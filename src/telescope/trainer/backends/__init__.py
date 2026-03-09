"""
Training backend abstraction.

Provides a unified interface for different distributed training strategies:
- FSDP (Fully Sharded Data Parallel) — PyTorch native, good for research/prototyping
- Megatron — NVIDIA Megatron-Core, good for tensor/pipeline parallelism at scale

Usage:
    from telescope.trainer.backends import create_backend
    backend = create_backend()
    backend.init()
    metrics = backend.train_step(trainer_data)
    state_dict = backend.gather_weights_for_inference()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class TrainingBackend(ABC):
    """
    Abstract base class for training backends.
    
    Each backend handles:
    - Distributed initialization (process groups, device mesh, etc.)
    - Model building (loading, wrapping with parallelism)
    - Optimizer creation
    - Training step (forward, loss, backward, optimizer)
    - Weight gathering for inference sync (in HF format)
    
    The backend is designed to be the same whether running in subprocess
    mode (current) or inside a Ray actor (future).
    """
    
    @abstractmethod
    def init(self) -> dict:
        """
        Initialize distributed groups, build model and optimizer.
        
        Must be called once before any other method.
        
        Returns:
            Dict with runtime info:
                - rank: int (global rank)
                - local_rank: int
                - world_size: int (total GPUs)
                - dp_rank: int (data parallel rank)
                - dp_world_size: int (number of data parallel replicas)
                - device: torch.device
        """
        ...

    @abstractmethod
    def train_step(self, trainer_data: dict) -> dict:
        """
        Execute one full RL training step.
        
        This includes: zero_grad, forward, loss, backward (for all micro batches),
        gradient clipping, and optimizer step.
        
        Args:
            trainer_data: Dict containing:
                - micro_batches: list of dicts, each with input_ids, loss_mask,
                  advantages, vllm_logprobs, position_ids as tensors
                - num_micro_batches: int
                
        Returns:
            Dict of metrics. Every key is automatically logged as a step
            metric to wandb.  Keys containing "/" are split into
            group/metric (e.g. "infra/tokens_trained" → group="infra",
            metric="tokens_trained").  Plain keys get group="general".
        """
        ...

    @abstractmethod
    def gather_weights_for_inference(self) -> dict[str, torch.Tensor]:
        """
        Gather full model state dict in HuggingFace format.
        
        ALL ranks must call this (it uses collective ops internally).
        Only the result on the designated rank (usually dp_rank=0, tp_rank=0)
        is meaningful for broadcasting.
        
        Returns:
            State dict mapping HF parameter names to full (unsharded) tensors.
            May be empty on non-source ranks.
        """
        ...

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all ranks in the training group."""
        ...
    
    @property
    @abstractmethod
    def rank(self) -> int:
        """Global rank in the training group."""
        ...

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """Local rank on this node."""
        ...

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Total number of GPUs in the training group."""
        ...

    @property
    @abstractmethod
    def dp_rank(self) -> int:
        """Data parallel rank (for data loading/distribution)."""
        ...

    @property
    @abstractmethod
    def dp_world_size(self) -> int:
        """Number of data parallel replicas."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Current CUDA device for this rank."""
        ...

    @property
    @abstractmethod
    def is_weight_broadcast_rank(self) -> bool:
        """Whether this rank is responsible for broadcasting weights to inference."""
        ...

    @abstractmethod
    def save_checkpoint(self, step: int, ckpt_dir: Path, tracker=None) -> None:
        """Save a distributed checkpoint. All ranks must call together."""
        ...

    @abstractmethod
    def load_checkpoint(self, step: int, ckpt_dir: Path) -> None:
        """Load a distributed checkpoint. All ranks must call together."""
        ...


def build_lr_scheduler(optimizer):
    """Build a LambdaLR learning rate scheduler from config.

    Shared by FSDP and Megatron backends. Returns None when no scheduler is
    needed (lr_scheduler='none', or 'constant' with warmup_steps=0).
    """
    import math

    from telescope.utils import config

    sched_type = config.cfg.lr_scheduler
    if sched_type == "none":
        return None

    warmup_steps = config.cfg.warmup_steps
    total_steps = config.cfg.number_of_steps
    min_lr_ratio = config.cfg.min_lr_ratio

    if sched_type == "constant":
        if warmup_steps == 0:
            return None  # constant with no warmup = no scheduler needed
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
    elif sched_type == "linear":
        decay_steps = max(total_steps - warmup_steps, 1)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = min((step - warmup_steps) / decay_steps, 1.0)
            return 1.0 - (1.0 - min_lr_ratio) * progress
    elif sched_type == "cosine":
        decay_steps = max(total_steps - warmup_steps, 1)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = min((step - warmup_steps) / decay_steps, 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_type!r}")

    from torch.optim.lr_scheduler import LambdaLR
    from telescope.utils.tlog import get_logger
    _log = get_logger("trainer")
    scheduler = LambdaLR(optimizer, lr_lambda)
    _log.info(
        f"LR scheduler: {sched_type}, warmup_steps={warmup_steps}, "
        f"total_steps={total_steps}, min_lr_ratio={min_lr_ratio}",
    )
    return scheduler


def create_backend() -> TrainingBackend:
    """
    Factory function to create the appropriate training backend.
    
    Reads TRAIN_BACKEND from config to determine which backend to use.
    
    Returns:
        TrainingBackend instance (not yet initialized — call .init() after).
    """
    from telescope.utils import config
    
    backend_name = config.cfg.train_backend
    
    if backend_name == "fsdp":
        from telescope.trainer.backends.fsdp import FSDPBackend
        return FSDPBackend()
    elif backend_name == "megatron":
        try:
            from telescope.trainer.backends.megatron import MegatronBackend
        except ImportError as e:
            raise ImportError(
                "Megatron backend requires megatron-core. "
                "Install with: uv sync --extra megatron"
            ) from e
        return MegatronBackend()
    else:
        raise ValueError(f"Unknown training backend: {backend_name!r}. Choose 'fsdp' or 'megatron'.")


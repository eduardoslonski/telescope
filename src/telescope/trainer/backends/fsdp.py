"""
FSDP (Fully Sharded Data Parallel) training backend.

Uses PyTorch native FSDP2 (fully_shard) for data parallelism with
optional mixed precision. Good for research/prototyping and models
that fit on a reasonable number of GPUs with data parallelism alone.

In FSDP mode:
- dp_rank == global_rank (every GPU is a data parallel replica)
- dp_world_size == world_size
- Model is sharded across all GPUs for memory efficiency
"""
from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from telescope.utils import config
from telescope.utils.tlog import get_logger, setup_logging
from telescope.trainer.backends import TrainingBackend, build_lr_scheduler
from telescope.trainer.loss import compute_rl_loss

if TYPE_CHECKING:
    from telescope.trainer.metrics.timeline import GPUTimelineLogger, _NullTracker

_log = get_logger("trainer")


def _get_torch_dtype(dtype_str: str | None) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    if dtype_str is None:
        return torch.float32
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


class FSDPBackend(TrainingBackend):
    """
    FSDP2 training backend using PyTorch native fully_shard.
    
    This is the default backend. It shards model parameters, gradients,
    and optimizer states across all GPUs. Every GPU holds a full copy
    of the data (data parallel), and FSDP handles the parameter sharding.
    """
    
    def __init__(self):
        self._rank = 0
        self._local_rank = 0
        self._world_size = 1
        self._device = torch.device("cpu")
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._tokenizer = None
        self._hf_config = None

    def init(self) -> dict:
        """Initialize FSDP: dist process group, model, optimizer."""
        setup_logging()
        
        dist.init_process_group(backend="nccl")
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._device = torch.device(f"cuda:{self._local_rank}")
        torch.cuda.set_device(self._local_rank)
        
        _log.banner("FSDP Backend Init")
        _log.info(
            f"world_size={self._world_size}, rank={self._rank}, "
            f"local_rank={self._local_rank}",
            rank=self._rank,
        )
        
        # Cache tokenizer + HF config for checkpoint metadata (parallel I/O)
        with ThreadPoolExecutor(max_workers=2) as pool:
            tok_future = pool.submit(AutoTokenizer.from_pretrained, config.cfg.model, trust_remote_code=True)
            cfg_future = pool.submit(AutoConfig.from_pretrained, config.cfg.model, trust_remote_code=True)
            self._tokenizer = tok_future.result()
            self._hf_config = cfg_future.result()

        # Build model with FSDP wrapping
        self._model = self._build_model()
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()

        return {
            "rank": self._rank,
            "local_rank": self._local_rank,
            "world_size": self._world_size,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "device": self._device,
            # FSDP is pure data parallel — no TP/PP/CP/EP.
            "tp_rank": 0,
            "tp_size": 1,
            "pp_rank": 0,
            "pp_size": 1,
            "cp_rank": 0,
            "cp_size": 1,
            "ep_rank": 0,
            "ep_size": 1,
        }

    def _build_model(self):
        """Load HuggingFace model and wrap with FSDP2."""
        model_dtype = _get_torch_dtype(config.cfg.model_dtype)
        # Prefer flash_attention_2 for correct packed-sequence training (varlen
        # kernel prevents cross-sample attention).  Fall back to sdpa if missing.
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            _log.warning(
                "flash-attn not installed — using PyTorch SDPA attention. "
                "Install for better performance: uv add flash-attn"
            )
        model = AutoModelForCausalLM.from_pretrained(
            config.cfg.model, torch_dtype=model_dtype, attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        if config.cfg.mixed_precision_dtype is not None:
            mp_dtype = _get_torch_dtype(config.cfg.mixed_precision_dtype)
            mp_policy = MixedPrecisionPolicy(param_dtype=mp_dtype, reduce_dtype=torch.float32)
        else:
            mp_policy = MixedPrecisionPolicy()

        # Use HF's _no_split_modules to discover transformer layer classes
        # (model-agnostic — works for Llama, Qwen2, Mistral, DeepSeek, etc.).
        # Also shard embeddings separately when embed_tokens and lm_head are
        # NOT tied, giving smaller all-gather groups and better overlap.
        layer_cls_names = getattr(model, "_no_split_modules", None)
        if layer_cls_names:
            modules_to_shard = [
                module for _, module in model.named_modules()
                if module.__class__.__name__ in layer_cls_names
                or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
            ]
            for module in modules_to_shard:
                fully_shard(module, mp_policy=mp_policy)
        else:
            _log.warning("Model does not define _no_split_modules, falling back to model.model.layers")
            for transformer_block in model.model.layers:
                fully_shard(transformer_block, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

        _log.section(f"Model Loaded: {config.cfg.model}")
        _log.info(f"FSDP wrapped with {config.cfg.mixed_precision_dtype or 'native dtype'} mixed precision")
        return model

    def _build_optimizer(self):
        """Build AdamW optimizer."""
        return torch.optim.AdamW(
            self._model.parameters(),
            lr=config.cfg.learning_rate,
            weight_decay=config.cfg.weight_decay,
        )

    def _build_scheduler(self):
        """Build learning rate scheduler (None if lr_scheduler='none')."""
        return build_lr_scheduler(self._optimizer)
    
    def _compute_batch_logprobs(self, micro_batches: list[dict]) -> None:
        """Compute logprobs with the current policy for all micro-batches (no grad).

        Stores ``batch_logprobs`` in each micro-batch dict, in the same shape as
        ``vllm_logprobs`` (``[batch, seq_len]`` with 0.0 at position 0).
        """
        self._model.eval()
        with torch.no_grad():
            for mb in micro_batches:
                input_ids = mb["input_ids"]
                if not input_ids.is_cuda:
                    input_ids = input_ids.to(self._device, non_blocking=True)
                position_ids = mb.get("position_ids")
                if position_ids is not None and not position_ids.is_cuda:
                    position_ids = position_ids.to(self._device, non_blocking=True)

                if position_ids is not None:
                    outputs = self._model(input_ids=input_ids, position_ids=position_ids)
                else:
                    outputs = self._model(input_ids=input_ids)

                logits = outputs.logits
                shift_logits = logits[..., :-1, :] / config.cfg.get_sampling_params()["temperature"]
                labels = input_ids[:, 1:]
                log_probs = shift_logits.log_softmax(dim=-1)
                log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

                # Pad to [batch, seq_len] with 0.0 at position 0 (same layout as vllm_logprobs)
                batch_logprobs = torch.zeros(
                    input_ids.shape[0], input_ids.shape[1],
                    device=log_probs.device, dtype=log_probs.dtype,
                )
                batch_logprobs[..., 1:] = log_probs
                mb["batch_logprobs"] = batch_logprobs
        self._model.train()

    def train_step(
        self,
        trainer_data: dict,
        tracker: GPUTimelineLogger | _NullTracker | None = None,
    ) -> dict:
        """
        Execute one RL training step with micro batching and optional minibatch groups.

        When ``NUMBER_OF_MINIBATCHES > 1``, micro-batches are split into that many
        groups, each getting its own zero_grad → forward/backward → grad_clip →
        optimizer.step() cycle.

        Timeline events emitted (when tracker provided):
        - zero_grad
        - data_to_device (per microbatch)
        - prepare_tensors (per microbatch)
        - forward (per microbatch)
        - loss_computation (per microbatch)
        - compute_entropy (per microbatch)
        - compute_kl (per microbatch)
        - backward (per microbatch)
        - grad_norm
        - grad_clip
        - optimizer
        """
        def _track(name: str, **kwargs):
            if tracker is not None:
                return tracker.track(name, **kwargs)
            return nullcontext()

        micro_batches = trainer_data["micro_batches"]

        # Compute batch logprobs if needed (before any weight updates).
        _needs_batch_logprobs = (
            config.cfg.ppo_clip_ref_logprobs == "batch"
            and (config.cfg.use_ppo_clip or config.cfg.algorithm in ("cispo", "gspo", "sapo"))
        )
        if _needs_batch_logprobs:
            self._compute_batch_logprobs(micro_batches)

        # Split micro-batches into minibatch groups.
        num_minibatches = config.cfg.number_of_minibatches
        groups: list[list[dict]] = [[] for _ in range(num_minibatches)]
        for i, mb in enumerate(micro_batches):
            groups[i % num_minibatches].append(mb)
        # Remove empty groups (when fewer micro-batches than minibatches).
        groups = [g for g in groups if g]

        # Accumulators for aggregated metrics across all minibatch groups.
        total_tokens = 0
        weighted_entropy = 0.0
        weighted_kl = 0.0
        all_grad_norms: list[float] = []

        use_minibatch = len(groups) > 1
        for group_idx, group in enumerate(groups):
            num_mb_in_group = len(group)
            mb_idx = group_idx if use_minibatch else -1

            with _track("zero_grad", minibatch=mb_idx):
                self._optimizer.zero_grad(set_to_none=True)

            group_tokens = 0
            group_entropy = 0.0
            group_kl = 0.0
            for micro_idx, micro_batch in enumerate(group):
                # Only reduce-scatter gradients on the last micro-batch.
                # Earlier micro-batches accumulate locally, avoiding redundant
                # NCCL communication during gradient accumulation.
                self._model.set_requires_gradient_sync(
                    micro_idx == num_mb_in_group - 1
                )
                mb_metrics = self._process_micro_batch(
                    micro_batch, num_mb_in_group, tracker=tracker, micro_idx=micro_idx,
                    minibatch=mb_idx,
                )
                nt = mb_metrics["num_tokens"]
                group_tokens += nt
                group_entropy += mb_metrics["entropy"] * nt
                group_kl += mb_metrics["kl_divergence_inference"] * nt

            # Gradient clipping
            with _track("grad_norm", minibatch=mb_idx):
                grad_norm = 0.0
                if config.cfg.grad_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), config.cfg.grad_clip
                    ).item()
                else:
                    total_norm_sq = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self._model.parameters()
                        if p.grad is not None
                    )
                    grad_norm = total_norm_sq ** 0.5

            # Optimizer step (skip on non-finite grad norm to avoid corrupting weights)
            with _track("optimizer", minibatch=mb_idx):
                if math.isfinite(grad_norm):
                    self._optimizer.step()
                else:
                    _log.warning(
                        f"Non-finite grad norm ({grad_norm}), skipping optimizer step",
                        rank=self._rank,
                    )
                    self._optimizer.zero_grad(set_to_none=True)

            all_grad_norms.append(grad_norm)
            total_tokens += group_tokens
            weighted_entropy += group_entropy
            weighted_kl += group_kl

        # Step the LR scheduler once per train_step (NOT per minibatch group).
        if self._scheduler is not None:
            self._scheduler.step()

        # Aggregate metrics (weighted avg by token count, grad_norm averaged).
        denom = max(total_tokens, 1)
        metrics = {
            "grad_norm": sum(all_grad_norms) / max(len(all_grad_norms), 1),
            "entropy": weighted_entropy / denom,
            "kl_divergence_inference": weighted_kl / denom,
        }
        metrics["learning_rate"] = (
            self._scheduler.get_last_lr()[0] if self._scheduler is not None
            else config.cfg.learning_rate
        )
        return metrics

    def _process_micro_batch(
        self,
        micro_batch: dict,
        num_micro_batches: int,
        tracker: GPUTimelineLogger | _NullTracker | None = None,
        micro_idx: int = 0,
        minibatch: int = -1,
    ) -> dict:
        """Process a single micro batch: forward, loss, backward."""
        def _track(name: str, **kwargs):
            if tracker is not None:
                return tracker.track(name, microbatch=micro_idx, minibatch=minibatch, **kwargs)
            return nullcontext()

        # Move to device
        with _track("data_to_device"):
            input_ids = micro_batch["input_ids"]
            if not input_ids.is_cuda:
                input_ids = input_ids.to(self._device, non_blocking=True)
            loss_mask = micro_batch["loss_mask"]
            if not loss_mask.is_cuda:
                loss_mask = loss_mask.to(self._device, non_blocking=True)
            advantages = micro_batch["advantages"]
            if not advantages.is_cuda:
                advantages = advantages.to(self._device, non_blocking=True)
            
            vllm_logprobs = micro_batch.get("vllm_logprobs")
            if vllm_logprobs is not None and not vllm_logprobs.is_cuda:
                vllm_logprobs = vllm_logprobs.to(self._device, non_blocking=True)
            
            position_ids = micro_batch.get("position_ids")
            if position_ids is not None and not position_ids.is_cuda:
                position_ids = position_ids.to(self._device, non_blocking=True)
        
        # Forward pass
        with _track("forward"):
            if position_ids is not None:
                outputs = self._model(input_ids=input_ids, position_ids=position_ids)
            else:
                outputs = self._model(input_ids=input_ids)
        
        # Determine ref_logprobs for ratio computation.
        ref_logprobs = None
        if config.cfg.ppo_clip_ref_logprobs == "batch":
            batch_lp = micro_batch.get("batch_logprobs")
            if batch_lp is not None:
                ref_logprobs = batch_lp if batch_lp.is_cuda else batch_lp.to(self._device, non_blocking=True)

        # Compute loss + fine-grained RL metrics with tracked sub-events.
        scaled_loss, metrics = compute_rl_loss(
            logits=outputs.logits,
            input_ids=input_ids,
            loss_mask=loss_mask,
            advantages=advantages,
            vllm_logprobs=vllm_logprobs,
            num_micro_batches=num_micro_batches,
            track=_track,
            position_ids=position_ids,
            ref_logprobs=ref_logprobs,
        )
        
        # Backward
        with _track("backward"):
            scaled_loss.backward()
        
        return metrics

    def gather_weights_for_inference(self) -> dict[str, torch.Tensor]:
        """Gather full FSDP state dict for broadcasting to inference."""
        opts = StateDictOptions(full_state_dict=True, cpu_offload=False)
        state_dict = get_model_state_dict(self._model, options=opts)
        return state_dict

    def barrier(self) -> None:
        """Synchronize all FSDP ranks."""
        dist.barrier()
    
    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def dp_rank(self) -> int:
        # In FSDP, every rank is a data parallel replica
        return self._rank

    @property
    def dp_world_size(self) -> int:
        # In FSDP, dp_world_size == world_size
        return self._world_size

    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def is_weight_broadcast_rank(self) -> bool:
        # In FSDP, rank 0 broadcasts weights
        return self._rank == 0

    @property
    def model(self):
        """Access the underlying model (for checkpointing etc)."""
        return self._model
    
    @property
    def optimizer(self):
        """Access the optimizer (for checkpointing etc)."""
        return self._optimizer

    @property
    def scheduler(self):
        """Access the LR scheduler (None if lr_scheduler='none')."""
        return self._scheduler

    def _get_rng_state(self) -> dict:
        """Collect per-rank RNG state for checkpoint."""
        return {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
        }

    def _load_rng_state(self, rng_state: dict) -> None:
        """Restore per-rank RNG state from checkpoint."""
        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])

    def save_checkpoint(self, step: int, ckpt_dir: Path, tracker=None) -> None:
        """Save FSDP checkpoint using PyTorch DCP. All ranks must call together."""
        import shutil

        def _track(name: str):
            if tracker is not None:
                return tracker.track(name, cpu=True)
            return nullcontext()

        tmp_path = ckpt_dir / f"step_{step}_tmp"
        final_path = ckpt_dir / f"step_{step}"

        with _track("mkdir_barrier"):
            if self._rank == 0:
                tmp_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()

        start = time.perf_counter()
        if self._rank == 0:
            _log.info(f"Saving FSDP checkpoint at step {step}", step=step, rank=self._rank)

        with _track("get_state_dict"):
            torch.cuda.synchronize()
            save_training_state = config.cfg.checkpoint_save_training_state
            if save_training_state:
                model_state, optim_state = get_state_dict(
                    self._model, [self._optimizer],
                    options=StateDictOptions(cpu_offload=True),
                )
                state_dict = {
                    "model": model_state,
                    "optimizer": optim_state,
                    "step": step,
                }
                if self._scheduler is not None:
                    state_dict["scheduler"] = self._scheduler.state_dict()
            else:
                model_state = get_model_state_dict(
                    self._model,
                    options=StateDictOptions(cpu_offload=True),
                )
                state_dict = {
                    "model": model_state,
                    "step": step,
                }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with _track("dcp_save"):
                    dcp_save(state_dict, checkpoint_id=str(tmp_path))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if self._rank == 0:
                        _log.warning(
                            f"Checkpoint save attempt {attempt + 1} failed: {e}. Retrying...",
                            step=step, rank=self._rank,
                        )
                    dist.barrier()
                    if self._rank == 0:
                        if tmp_path.exists():
                            shutil.rmtree(tmp_path)
                        tmp_path.mkdir(parents=True, exist_ok=True)
                    dist.barrier()
                else:
                    raise

        # Save HF metadata (config + tokenizer) for offline conversion
        with _track("save_hf_meta"):
            if self._rank == 0:
                import json

                hf_meta_dir = tmp_path / "hf_meta"
                hf_meta_dir.mkdir(exist_ok=True)
                self._hf_config.save_pretrained(hf_meta_dir)
                self._tokenizer.save_pretrained(hf_meta_dir)

                meta = {
                    "base_model": config.cfg.model,
                    "step": step,
                    "backend": "fsdp",
                }
                meta_path = tmp_path / "meta.json"
                meta_path.write_text(json.dumps(meta, indent=2))

        # Save per-rank RNG state for deterministic resume
        if save_training_state:
            with _track("save_rng_state"):
                rng_path = tmp_path / f"rng_state_rank_{self._rank}.pt"
                torch.save(self._get_rng_state(), rng_path)

        # Pseudo-atomic: rank 0 renames tmp -> final after all ranks finish writing
        with _track("barrier_rename"):
            dist.barrier()
            if self._rank == 0:
                if final_path.exists():
                    shutil.rmtree(final_path)
                tmp_path.rename(final_path)
            dist.barrier()

        elapsed = time.perf_counter() - start
        if self._rank == 0:
            _log.timing("FSDP checkpoint saved", elapsed, step=step, rank=self._rank)

    def load_checkpoint(self, step: int, ckpt_dir: Path) -> None:
        """Load FSDP checkpoint using PyTorch DCP. All ranks must call together."""
        ckpt_path = ckpt_dir / f"step_{step}"

        start = time.perf_counter()
        if self._rank == 0:
            _log.info(f"Loading FSDP checkpoint from step {step}", step=step, rank=self._rank)

        # Build state dict structure matching what was saved
        torch.cuda.synchronize()
        model_state, optim_state = get_state_dict(
            self._model, [self._optimizer],
            options=StateDictOptions(cpu_offload=True),
        )
        state_dict = {
            "model": model_state,
            "optimizer": optim_state,
            "step": 0,
        }

        # Load from DCP checkpoint
        dcp_load(state_dict, checkpoint_id=str(ckpt_path))

        # Apply loaded state back to model and optimizer
        set_state_dict(
            self._model, [self._optimizer],
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=StateDictOptions(cpu_offload=True),
        )

        # Restore scheduler state if present
        if self._scheduler is not None and "scheduler" in state_dict:
            self._scheduler.load_state_dict(state_dict["scheduler"])

        # Restore per-rank RNG state
        rng_path = ckpt_path / f"rng_state_rank_{self._rank}.pt"
        rng_state = torch.load(rng_path, weights_only=False)
        self._load_rng_state(rng_state)

        dist.barrier()
        elapsed = time.perf_counter() - start
        if self._rank == 0:
            _log.timing("FSDP checkpoint loaded", elapsed, step=step, rank=self._rank)


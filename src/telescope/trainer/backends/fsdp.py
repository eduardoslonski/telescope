"""
FSDP (Fully Sharded Data Parallel) training backend.

Uses PyTorch native FSDP2 (fully_shard) for data parallelism with
optional mixed precision and optional context parallelism (ring attention).

When ``fsdp_context_parallel_size > 1``, a 2D device mesh ``(dp, cp)`` is
created and flattened into a single FSDP shard dimension so that CP ranks
also participate in parameter sharding (same approach as prime-rl / Composer 2).
Ring attention is enabled via ``ring-flash-attn`` which monkey-patches
HuggingFace flash attention kernels.
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
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from telescope.trainer.context_parallel import setup_cp_params, shard_for_cp

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
    """FSDP2 training backend using PyTorch native fully_shard.

    When ``fsdp_context_parallel_size > 1``, ring attention is used to
    distribute the sequence dimension across CP ranks while FSDP still
    shards parameters across *all* ranks (DP + CP folded together).
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
        # Context parallelism (ring attention)
        self._cp_size = 1
        self._cp_rank = 0
        self._cp_group: dist.ProcessGroup | None = None
        self._cp_mesh = None
        self._fsdp_mesh = None

    def init(self) -> dict:
        """Initialize FSDP: dist process group, model, optimizer."""
        setup_logging()

        dist.init_process_group(backend="nccl")
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._device = torch.device(f"cuda:{self._local_rank}")
        torch.cuda.set_device(self._local_rank)

        # --- Context parallelism setup ---
        self._cp_size = max(1, config.cfg.fsdp_context_parallel_size)
        if self._cp_size > 1:
            if self._world_size % self._cp_size != 0:
                raise ValueError(
                    f"WORLD_SIZE ({self._world_size}) must be divisible by "
                    f"fsdp_context_parallel_size ({self._cp_size})"
                )
            dp_size = self._world_size // self._cp_size
            # 2D mesh: (dp_shard, cp).  FSDP shards across both dimensions
            # (folded into one) so CP ranks also participate in param sharding.
            self._cp_mesh = init_device_mesh(
                "cuda",
                (dp_size, self._cp_size),
                mesh_dim_names=("dp", "cp"),
            )
            self._cp_group = self._cp_mesh.get_group("cp")
            self._cp_rank = self._cp_mesh.get_local_rank("cp")

            # Flatten (dp, cp) into a single FSDP shard dimension.
            self._fsdp_mesh = self._cp_mesh._flatten("dp_shard_cp")
        else:
            self._fsdp_mesh = None  # Use default process group

        _log.banner("FSDP Backend Init")
        _log.info(
            f"world_size={self._world_size}, rank={self._rank}, "
            f"local_rank={self._local_rank}, cp_size={self._cp_size}, "
            f"cp_rank={self._cp_rank}, dp_world_size={self.dp_world_size}",
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

        # Patch attention for ring attention after model is built.
        if self._cp_size > 1:
            from ring_flash_attn import substitute_hf_flash_attn
            substitute_hf_flash_attn(self._cp_group, heads_k_stride=1)
            _log.info(
                f"Ring attention enabled: cp_size={self._cp_size}",
                rank=self._rank,
            )

        return {
            "rank": self._rank,
            "local_rank": self._local_rank,
            "world_size": self._world_size,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "device": self._device,
            "tp_rank": 0,
            "tp_size": 1,
            "pp_rank": 0,
            "pp_size": 1,
            "cp_rank": self._cp_rank,
            "cp_size": self._cp_size,
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
            if self._cp_size > 1:
                raise ImportError(
                    "flash-attn is required for ring attention (fsdp_context_parallel_size > 1). "
                    "Install it: uv add flash-attn"
                )
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

        # When CP > 1 the flattened (dp, cp) mesh is passed so FSDP shards
        # across both DP *and* CP ranks (prime-rl / Composer 2 approach).
        fsdp_kwargs: dict = {"mp_policy": mp_policy}
        if self._fsdp_mesh is not None:
            fsdp_kwargs["mesh"] = self._fsdp_mesh

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
                fully_shard(module, **fsdp_kwargs)
        else:
            _log.warning("Model does not define _no_split_modules, falling back to model.model.layers")
            for transformer_block in model.model.layers:
                fully_shard(transformer_block, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

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

        With CP the forward runs on sharded sequences but the stored logprobs
        retain the *full* sequence length so that downstream sharding in
        ``_process_micro_batch`` works identically.
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

                full_seq_len = input_ids.shape[1]

                # Shard for CP before forward (same as _process_micro_batch).
                if self._cp_size > 1 and position_ids is not None:
                    input_ids, position_ids = setup_cp_params(
                        input_ids, position_ids,
                        self._cp_rank, self._cp_size, self._cp_group,
                    )

                if position_ids is not None:
                    outputs = self._model(input_ids=input_ids, position_ids=position_ids)
                else:
                    outputs = self._model(input_ids=input_ids)

                logits = outputs.logits
                shift_logits = logits[..., :-1, :] / config.cfg.get_sampling_params()["temperature"]
                labels = input_ids[:, 1:]
                log_probs = shift_logits.log_softmax(dim=-1)
                log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

                local_len = input_ids.shape[1]
                local_logprobs = torch.zeros(
                    input_ids.shape[0], local_len,
                    device=log_probs.device, dtype=log_probs.dtype,
                )
                local_logprobs[..., 1:] = log_probs

                if self._cp_size > 1:
                    # Gather chunks from all CP ranks → full-sequence logprobs.
                    chunks = [torch.zeros_like(local_logprobs) for _ in range(self._cp_size)]
                    dist.all_gather(chunks, local_logprobs, group=self._cp_group)
                    mb["batch_logprobs"] = torch.cat(chunks, dim=1)
                else:
                    mb["batch_logprobs"] = local_logprobs
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
        # Accumulate extra per-microbatch metrics (token-weighted like entropy/KL).
        _extra_keys: set[str] = set()
        _extra_weighted: dict[str, float] = {}
        _CORE_KEYS = {"loss", "entropy", "kl_divergence_inference", "num_tokens"}

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
                # Accumulate any extra metrics from new stability features
                for k, v in mb_metrics.items():
                    if k not in _CORE_KEYS:
                        _extra_keys.add(k)
                        _extra_weighted[k] = _extra_weighted.get(k, 0.0) + v * nt

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
        for k in _extra_keys:
            metrics[k] = _extra_weighted[k] / denom
        metrics["learning_rate"] = (
            self._scheduler.get_last_lr()[0] if self._scheduler is not None
            else config.cfg.learning_rate
        )
        # Pass through filter stats from batch preprocessing (if present)
        filter_stats = trainer_data.get("filter_stats")
        if filter_stats:
            metrics.update(filter_stats)
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

        # --- Context parallelism: shard all per-token tensors ---
        if self._cp_size > 1 and position_ids is not None:
            # setup_cp_params registers cu_seqlens with the ring kernel
            # (must be called before forward) and shards input_ids + position_ids.
            input_ids, position_ids = setup_cp_params(
                input_ids, position_ids,
                self._cp_rank, self._cp_size, self._cp_group,
            )
            loss_mask = shard_for_cp(loss_mask, self._cp_rank, self._cp_size)
            advantages = shard_for_cp(advantages, self._cp_rank, self._cp_size)
            if vllm_logprobs is not None:
                vllm_logprobs = shard_for_cp(vllm_logprobs, self._cp_rank, self._cp_size)

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
                if self._cp_size > 1:
                    ref_logprobs = shard_for_cp(ref_logprobs, self._cp_rank, self._cp_size)

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
        # With CP, multiple ranks share the same data — dp_rank groups them.
        if self._cp_mesh is not None:
            return self._cp_mesh.get_local_rank("dp")
        return self._rank

    @property
    def dp_world_size(self) -> int:
        # With CP, dp_world_size is the number of independent data shards.
        return self._world_size // self._cp_size

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


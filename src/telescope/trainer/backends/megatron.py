"""
Megatron-Core training backend.

Uses NVIDIA Megatron-Core for tensor parallelism (TP), pipeline parallelism (PP),
and context parallelism (CP). This enables training models that are too large to
fit on a single GPU (14B+) by sharding the model across multiple GPUs.

Architecture:
    With TP=2, PP=1, CP=1 on 4 GPUs:
        - GPUs 0,1 form TP group 0 (dp_rank=0)
        - GPUs 2,3 form TP group 1 (dp_rank=1)
        - dp_world_size = world_size / (TP * PP * CP) = 4 / 2 = 2

Weight sync to inference:
    1. All-gather TP shards to reconstruct full tensors
    2. Convert Megatron parameter names to HuggingFace format
    3. Broadcast via NCCL to vLLM inference servers (same as FSDP)

Dependencies:
    Only uses megatron.core (uv add megatron-core megatron-bridge). Does NOT require
    the full Megatron-LM repository. Builds TransformerConfig directly from HuggingFace
    model config, bypassing megatron.training entirely.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
import random
import re
import sys
import time
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from telescope.utils import config
from telescope.utils.tlog import get_logger, setup_logging
from telescope.trainer.backends import TrainingBackend, build_lr_scheduler

if TYPE_CHECKING:
    from telescope.trainer.metrics.timeline import GPUTimelineLogger, _NullTracker

_log = get_logger("trainer")
_megatron_logger = logging.getLogger("megatron")
_megatron_logger.setLevel(logging.WARNING)


# ============================================================================
# Utility: vocab size padding
# ============================================================================

def _pad_vocab_size(vocab_size: int, tp_size: int, pad_to_multiple: int = 64) -> int:
    """
    Pad vocab_size to be divisible by both tp_size and pad_to_multiple.
    
    This ensures tensor parallel sharding works correctly and improves
    CUDA kernel efficiency via alignment.
    
    """
    alignment = math.lcm(tp_size, pad_to_multiple)
    return math.ceil(vocab_size / alignment) * alignment


def _torch_dtype_from_config(value, default: torch.dtype) -> torch.dtype:
    """
    Parse a dtype config value into torch.dtype.

    Accepts torch.dtype directly or a torch dtype attribute name string
    (e.g., "float32", "bfloat16").
    """
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        if not hasattr(torch, value):
            raise ValueError(f"Unknown torch dtype string: {value!r}")
        dtype = getattr(torch, value)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Config value {value!r} is not a torch dtype")
        return dtype
    raise TypeError(f"Unsupported dtype config value type: {type(value)!r}")


def _maybe_install_unified_memory_stub() -> bool:
    """
    Optionally bypass Megatron import-time unified-memory JIT compilation.

    In some environments the import path
    `megatron.core.inference.unified_memory -> torch.cpp_extension.load_inline`
    can block for minutes on file-baton compile locks and make trainer startup
    look stuck. Telescope's trainer path does not use Megatron inference contexts,
    so stubbing this module is safe.

    Enable/disable with either:
    - config.cfg.megatron_disable_unified_memory_jit (defaults to True if missing)
    - env TELESCOPE_MEGATRON_DISABLE_UNIFIED_MEMORY_JIT (overrides config)
    """
    env_override = os.environ.get("TELESCOPE_MEGATRON_DISABLE_UNIFIED_MEMORY_JIT")
    if env_override is None:
        disable = bool(config.cfg.megatron_disable_unified_memory_jit)
    else:
        disable = env_override.strip().lower() not in {"0", "false", "no", "off"}

    if not disable:
        return False

    module_name = "megatron.core.inference.unified_memory"
    if module_name in sys.modules:
        return False

    stub = types.ModuleType(module_name)
    stub.has_unified_memory = False

    def _disabled_create_unified_mempool():
        raise RuntimeError("Unified memory mempool is disabled in telescope trainer.")

    stub.create_unified_mempool = _disabled_create_unified_mempool

    # Newer megatron-core versions import these exceptions from unified_memory
    # in megatron.core.inference.contexts.dynamic_context.
    class UnifiedMemoryUnsupportedError(Exception):
        """Unified memory is not supported on this system."""

    class UnifiedMemoryCompileTimeoutError(UnifiedMemoryUnsupportedError):
        """Unified memory compilation timed out."""

    stub.UnifiedMemoryUnsupportedError = UnifiedMemoryUnsupportedError
    stub.UnifiedMemoryCompileTimeoutError = UnifiedMemoryCompileTimeoutError

    sys.modules[module_name] = stub
    return True


def _build_packed_segment_attention_mask_from_position_ids(
    position_ids: torch.Tensor | None,
):
    """
    Build a segment mask from non-monotonic position_ids.

    Telescope can produce non-monotonic position_ids (multi-turn packed prefixes).
    DotProductAttention does not support PackedSeqParams, so we explicitly mask
    cross-segment attention while keeping causal masking in Megatron.
    """
    if position_ids is None:
        return None
    if position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    if position_ids.numel() <= 1:
        return None

    pos = position_ids[0]
    # Segment boundaries occur where the position increment is not +1.
    first_dummy = pos[:1] - 1
    pos_diff = torch.diff(pos, prepend=first_dummy, dim=0)
    seg_ids = (pos_diff != 1).cumsum(dim=0)  # [seq]

    # If all tokens are in one segment, Megatron's built-in causal mask is sufficient.
    if int(seg_ids[-1].item()) == 0:
        return None

    seq_len = int(pos.numel())
    # True means "masked out" in Megatron attention_mask_func.
    seg_mask = seg_ids.unsqueeze(0).unsqueeze(-1) != seg_ids.unsqueeze(0).unsqueeze(-2)  # [1, s, s]
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=pos.device),
        diagonal=1,
    ).unsqueeze(0)  # [1, s, s]
    full_mask = seg_mask | causal_mask
    return full_mask.unsqueeze(1)  # [1, 1, s, s]


def _vocab_parallel_log_probs_and_entropy(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute selected-token logprobs and entropy from TP-sharded vocab logits.

    Labels are global token IDs while logits are local vocab shards on each TP rank.
    """
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        log_probs_full = torch.log_softmax(vocab_parallel_logits.float(), dim=-1)
        log_probs = torch.gather(
            log_probs_full,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        entropy = -(log_probs_full.exp() * log_probs_full).sum(dim=-1)
        return log_probs, entropy

    tp_rank = dist.get_rank(tp_group)
    partition_vocab_size = vocab_parallel_logits.size(-1)
    vocab_start = tp_rank * partition_vocab_size
    vocab_end = vocab_start + partition_vocab_size

    logits = vocab_parallel_logits.float()

    # Shared log-sum-exp denominator across TP ranks.
    logits_max = logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

    normalized = logits - logits_max
    exp_logits = normalized.exp()
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
    log_sum_exp = sum_exp.log()

    # Gather selected logits from the owning shard.
    labels_mask = (labels < vocab_start) | (labels >= vocab_end)
    masked_labels = labels.clone() - vocab_start
    masked_labels[labels_mask] = 0

    local_selected = torch.gather(
        normalized,
        dim=-1,
        index=masked_labels.unsqueeze(-1),
    ).squeeze(-1)
    local_selected[labels_mask] = 0.0
    dist.all_reduce(local_selected, op=dist.ReduceOp.SUM, group=tp_group)

    log_probs = local_selected - log_sum_exp.squeeze(-1)

    # Entropy over global vocab from TP shards.
    softmax = exp_logits / sum_exp
    local_softmax_times_logits = (softmax * logits).sum(dim=-1)
    dist.all_reduce(local_softmax_times_logits, op=dist.ReduceOp.SUM, group=tp_group)
    entropy = logits_max.squeeze(-1) + log_sum_exp.squeeze(-1) - local_softmax_times_logits

    return log_probs, entropy


def _compute_rl_loss_vocab_parallel(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
    vllm_logprobs: torch.Tensor | None,
    tp_group,
    num_micro_batches: int,
    track: Callable[[str], object] | None = None,
    position_ids: torch.Tensor | None = None,
    ref_logprobs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    TP-safe RL loss path for Megatron when logits are vocab-parallel shards.

    Args:
        ref_logprobs: Reference logprobs for ratio computation (use_ppo_clip, cispo, gspo, sapo).
            When None, falls back to vllm_logprobs. vllm_logprobs is still used for TIS and metrics.
    """
    def _track(event_name: str):
        if track is not None:
            return track(event_name)
        return nullcontext()

    with _track("prepare_tensors"):
        if advantages.dim() == 2:
            advantages = advantages.unsqueeze(-1)

        shift_logits = logits[..., :-1, :] / config.cfg.get_sampling_params()["temperature"]
        labels = input_ids[:, 1:]
        shift_loss_mask = loss_mask[..., 1:]
        shift_advantages = advantages[..., 1:, :]

    with _track("loss_computation"):
        log_probs, entropy = _vocab_parallel_log_probs_and_entropy(
            shift_logits,
            labels,
            tp_group,
        )

        # Delta for ratio computation (use_ppo_clip, cispo, gspo, sapo).
        # Uses ref_logprobs when provided, otherwise falls back to vllm_logprobs.
        _ref = ref_logprobs if ref_logprobs is not None else vllm_logprobs
        ratio_delta = None
        if _ref is not None:
            ratio_delta = log_probs - _ref[..., 1:]

        # Delta against rollout (vLLM) logprobs — always used for TIS and metrics.
        vllm_delta = None
        if vllm_logprobs is not None:
            vllm_delta = log_probs - vllm_logprobs[..., 1:]

        # --- Policy loss (algorithm dispatch) ---
        algo = config.cfg.algorithm
        adv = shift_advantages.squeeze(-1)
        _needs_ratio = algo in ("cispo", "gspo", "sapo")

        if _needs_ratio and ratio_delta is None:
            raise ValueError(
                f"ALGORITHM={algo!r} requires reference logprobs (vllm_logprobs or ref_logprobs) "
                f"but none were provided"
            )

        if algo == "cispo":
            ratio = torch.exp(ratio_delta.clamp(-20.0, 20.0))
            clamped = torch.clamp(ratio, 1.0 - config.cfg.clip_low, 1.0 + config.cfg.clip_high)
            policy_loss = -clamped.detach() * log_probs * adv

        elif algo == "gspo":
            if position_ids is None:
                raise ValueError("GSPO requires position_ids for sequence boundary detection in packed batches")
            from telescope.trainer.loss import _masked_mean_per_sample, _masked_count_per_sample
            log_delta = ratio_delta.unsqueeze(0) if ratio_delta.dim() == 1 else ratio_delta
            mask_2d = shift_loss_mask.unsqueeze(0) if shift_loss_mask.dim() == 1 else shift_loss_mask

            # Sequence-level geometric mean of log ratios (detached, for value)
            log_seq_ratio = _masked_mean_per_sample(log_delta, mask_2d, position_ids)

            # Per-token count of valid tokens in each sample (for 1/N gradient scaling)
            token_counts = _masked_count_per_sample(mask_2d, position_ids).squeeze(0)

            # Gradient trick with 1/seq_len scaling for geometric mean:
            #   value  = log_seq_ratio (sequence-level mean)
            #   grad/token = 1/count  (not full 1.0)
            scaled_delta = ratio_delta / token_counts.clamp_min(1)
            log_token_ratio = scaled_delta - scaled_delta.detach() + log_seq_ratio.squeeze(0).detach()
            ratio = torch.exp(log_token_ratio.clamp(max=10.0))

            # Average advantages per sequence (keeps gradient magnitude independent of seq length)
            adv_2d = adv.unsqueeze(0) if adv.dim() == 1 else adv
            adv = _masked_mean_per_sample(adv_2d, mask_2d, position_ids).squeeze(0)

            pg_loss1 = -ratio * adv
            clipped_ratio = torch.clamp(ratio, 1.0 - config.cfg.clip_low, 1.0 + config.cfg.clip_high)
            pg_loss2 = -clipped_ratio * adv
            policy_loss = torch.max(pg_loss1, pg_loss2)

        elif algo == "sapo":
            ratio = torch.exp(ratio_delta.clamp(-20.0, 20.0))
            tau = torch.where(adv > 0, config.cfg.sapo_tau_pos, config.cfg.sapo_tau_neg)
            gate = torch.sigmoid(tau * (ratio - 1.0)) * (4.0 / tau)
            policy_loss = -gate * adv

        elif algo in ("grpo", "rloo", "reinforce_pp", "dr_grpo"):
            policy_loss = -log_probs * adv

        else:
            raise ValueError(f"Unknown ALGORITHM: {algo!r}")

        # --- PPO clip (algorithm-agnostic trust-region clipping) ---
        if config.cfg.use_ppo_clip:
            if ratio_delta is None:
                raise ValueError(
                    "USE_PPO_CLIP requires reference logprobs (vllm_logprobs or ref_logprobs) "
                    "but none were provided"
                )
            ratio = torch.exp(ratio_delta.clamp(-20.0, 20.0))
            pg_loss1 = -ratio * adv
            clipped_ratio = torch.clamp(ratio, 1.0 - config.cfg.clip_low, 1.0 + config.cfg.clip_high)
            pg_loss2 = -clipped_ratio * adv
            policy_loss = torch.max(pg_loss1, pg_loss2)

        # --- TIS correction (always uses vllm_delta = distance from rollout policy) ---
        if config.cfg.use_tis and vllm_delta is not None:
            logprob_diff = vllm_delta.clamp(
                -config.cfg.tis_logprob_clamp, config.cfg.tis_logprob_clamp
            )
            tis_ratio = torch.exp(logprob_diff).clamp(max=config.cfg.tis_cap).detach()
            policy_loss = policy_loss * tis_ratio

        policy_loss = policy_loss * shift_loss_mask
        num_tokens = int(shift_loss_mask.sum().item())
        token_denom = shift_loss_mask.sum().clamp_min(1).float()

        if algo == "dr_grpo" and config.cfg.dr_grpo_loss_agg_mode == "token_sum_norm":
            if position_ids is not None:
                starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0]
                num_samples = len(starts)
                # Padding appended by pad_micro_batch starts with position_id=0,
                # creating a ghost segment with no trainable tokens.  Exclude it.
                if num_samples > 1 and not loss_mask[0, starts[-1]:].any():
                    num_samples -= 1
            else:
                num_samples = 1
            denom_val = float(num_samples * shift_loss_mask.shape[-1])
            loss = policy_loss.sum() / max(denom_val, 1.0)
        else:
            loss = policy_loss.sum() / token_denom
        scaled_loss = loss / num_micro_batches

    with _track("compute_entropy"):
        entropy_value = float(((entropy * shift_loss_mask).sum() / token_denom).item())

    with _track("compute_kl"):
        kl_value = 0.0
        if vllm_delta is not None:
            from telescope.trainer.loss import _compute_kl_from_logprob_diff
            kl_value = _compute_kl_from_logprob_diff(vllm_delta, shift_loss_mask)

    metrics = {
        "loss": float(loss.item()),
        "entropy": entropy_value,
        "kl_divergence_inference": kl_value,
        "num_tokens": num_tokens,
    }
    return scaled_loss, metrics


def _can_use_sequence_parallel(normalization: str = "RMSNorm") -> bool:
    """
    Return whether sequence parallel is safe with the current layer spec.

    With TransformerEngine spec, sequence parallel works natively (TE has its
    own fused norms).  With the local spec, sequence parallel requires Apex's
    FusedLayerNorm — but FusedLayerNorm only supports LayerNorm, not RMSNorm.
    Since virtually all modern models use RMSNorm, the local spec path falls
    back to WrappedTorchNorm which asserts ``not sequence_parallel``.
    """
    if config.cfg.megatron_use_transformer_engine:
        try:
            import transformer_engine  # noqa: F401
            return True
        except ImportError:
            return False
    # Local spec: RMSNorm always uses WrappedTorchNorm which doesn't support SP.
    # Only LayerNorm can use Apex's FusedLayerNorm with SP.
    if normalization != "LayerNorm":
        return False
    try:
        import apex  # noqa: F401
    except Exception:
        return False
    return True


# ============================================================================
# HF config → Megatron TransformerConfig conversion
# ============================================================================

def _hf_to_transformer_config(
    hf_config,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    ep_size: int,
    dtype: torch.dtype = torch.bfloat16,
    seq_parallel: bool = True,
    gradient_checkpointing: bool = True,
) -> "TransformerConfig":
    """
    Build a Megatron TransformerConfig directly from a HuggingFace model config.

    Bypasses megatron.training.arguments entirely — only requires megatron.core.
    """
    from megatron.core.transformer import TransformerConfig

    # Detect model-specific settings from HF config
    model_type = getattr(hf_config, "model_type", "").lower()
    # Qwen2 checkpoints can include q/k/v projection biases even when
    # the config object (older Transformers versions) does not expose
    # an `attention_bias` field. If we miss this, Megatron builds
    # bias-less attention and silently drops those bias tensors.
    attention_bias = getattr(hf_config, "attention_bias", None)
    if attention_bias is None:
        has_qkv_bias = model_type.startswith("qwen2")
    else:
        has_qkv_bias = bool(attention_bias)
    has_qk_norm = getattr(hf_config, "qk_layernorm", False) or "qwen3" in model_type

    config_kwargs = {
        # Model architecture (from HF config)
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        "ffn_hidden_size": hf_config.intermediate_size,
        "kv_channels": getattr(hf_config, "head_dim", None),
        "rotary_base": int(getattr(hf_config, "rope_theta", 10000)),
        "layernorm_epsilon": getattr(hf_config, "rms_norm_eps", 1e-6),
        # Activation and normalization
        "activation_func": F.silu,
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        # Bias
        "add_bias_linear": False,
        "add_qkv_bias": has_qkv_bias,
        # QK layernorm (Qwen3)
        "qk_layernorm": has_qk_norm,
        # Dropout (disabled for RL training)
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        # Data types
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        "fp16": dtype is torch.float16,
        # Parallelism
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        "context_parallel_size": cp_size,
        "expert_model_parallel_size": ep_size,
        "expert_tensor_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "sequence_parallel": seq_parallel and tp_size > 1,
        # Pipeline parallel comm
        "overlap_p2p_comm": False,
        "batch_p2p_comm": False,
        # Common settings
        "variable_seq_lengths": True,
        "moe_token_dispatcher_type": "alltoall",  # Required when variable_seq_lengths=True
        "masked_softmax_fusion": False,
    }

    # ---- MoE fields (present only for Mixture-of-Experts models) ----
    # HF config field names vary by model family.
    num_moe_experts = getattr(hf_config, "num_experts", None)  # Qwen2MoE / Qwen3MoE
    if num_moe_experts is None:
        num_moe_experts = getattr(hf_config, "num_local_experts", None)  # Mixtral / DeepSeek

    if num_moe_experts is not None:
        config_kwargs["num_moe_experts"] = num_moe_experts
        config_kwargs["moe_router_topk"] = getattr(hf_config, "num_experts_per_tok", 2)
        moe_ffn = getattr(hf_config, "moe_intermediate_size", None)
        if moe_ffn is not None:
            config_kwargs["moe_ffn_hidden_size"] = moe_ffn
        shared = getattr(hf_config, "shared_expert_intermediate_size", None)
        if shared is not None:
            config_kwargs["moe_shared_expert_intermediate_size"] = shared

        # Router scoring behaviour must match the original model architecture.
        # - scoring_func is exposed by DeepSeek configs ("sigmoid"); most others
        #   default to "softmax".
        # - pre_softmax controls whether softmax is applied before top-k.
        #   Qwen2MoE / Mixtral use pre-softmax routing; Qwen3MoE does not.
        scoring_func = getattr(hf_config, "scoring_func", None)
        if scoring_func is not None:
            config_kwargs["moe_router_score_function"] = scoring_func
        # Qwen2MoE and Mixtral use pre-softmax routing, Qwen3MoE does not.
        if "qwen2" in model_type:      # qwen2_moe
            config_kwargs["moe_router_pre_softmax"] = True
        elif "qwen3" in model_type:     # qwen3_moe
            config_kwargs["moe_router_pre_softmax"] = False
        elif "mixtral" in model_type:
            config_kwargs["moe_router_pre_softmax"] = True

    config_kwargs.update({
        # Initialization
        "use_cpu_initialization": False,
        "perform_initialization": True,
    })

    # Gradient checkpointing / recomputation
    if gradient_checkpointing:
        config_kwargs["recompute_granularity"] = "full"
        config_kwargs["recompute_method"] = "uniform"
        config_kwargs["recompute_num_layers"] = 1

    # FP8 (requires TransformerEngine + Hopper GPU)
    if config.cfg.megatron_use_transformer_engine and config.cfg.megatron_fp8:
        config_kwargs["fp8"] = "e4m3"
        config_kwargs["fp8_margin"] = 0
        config_kwargs["fp8_interval"] = 1
        config_kwargs["fp8_amax_history_len"] = 1024
        config_kwargs["fp8_amax_compute_algo"] = "max"

    # Filter out keys not supported in the current megatron-core version
    supported_keys = {f.name for f in dataclasses.fields(TransformerConfig)}
    unsupported = [k for k in config_kwargs if k not in supported_keys]
    if unsupported:
        _log.info(f"Removing unsupported TransformerConfig keys: {unsupported}")
        for k in unsupported:
            config_kwargs.pop(k)

    return TransformerConfig(**config_kwargs)


# ============================================================================
# Megatron → HuggingFace weight name conversion
# ============================================================================
# These converters are model-specific. Add new ones for other architectures.

def _convert_qwen_to_hf(
    name: str,
    param: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    hidden_size: int,
    head_dim: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """
    Convert a single Megatron parameter (name + tensor) to HuggingFace format.
    
    Handles Qwen2 and Qwen3 models (same transformer architecture).
    
    Returns list of (hf_name, hf_tensor) tuples. May return multiple entries
    for fused parameters (e.g., QKV → separate Q, K, V).
    """
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    value_num_per_group = num_attention_heads // num_key_value_heads

    # Embedding and output
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    # Decoder layers
    decoder_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.weight":
            # Megatron stores fused QKV: [num_groups * (num_heads_per_group + 2) * head_dim, hidden]
            # Split into separate Q, K, V
            param = param.view(num_key_value_heads, -1, head_dim, hidden_size)
            q_param, k_param, v_param = torch.split(
                param, [value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, hidden_size)
            k_param = k_param.reshape(-1, hidden_size)
            v_param = v_param.reshape(-1, hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]

        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(num_key_value_heads, -1)
            q_bias, k_bias, v_bias = torch.split(
                param, [value_num_per_group * head_dim, head_dim, head_dim], dim=1
            )
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias.contiguous().flatten()),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias.contiguous().flatten()),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias.contiguous().flatten()),
            ]

        elif rest == "mlp.linear_fc1.weight":
            # Megatron fuses gate + up projections
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]

        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

        # MoE router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]

        # MoE expert weights (local_experts)
        elif rest.startswith("mlp.experts.local_experts."):
            expert_match = re.match(
                r"mlp\.experts\.local_experts\.(\d+)\.(.+)", rest
            )
            if expert_match:
                expert_idx, expert_rest = expert_match.groups()
                if expert_rest == "linear_fc1.weight":
                    gate_w, up_w = param.chunk(2, dim=0)
                    return [
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_w),
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_w),
                    ]
                elif expert_rest == "linear_fc2.weight":
                    return [(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param)]

        # MoE shared expert weights (Qwen2MoE, DeepSeek)
        elif rest.startswith("mlp.shared_experts."):
            if rest == "mlp.shared_experts.linear_fc1.weight":
                gate_w, up_w = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_w),
                    (f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_w),
                ]
            elif rest == "mlp.shared_experts.linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "mlp.shared_experts.gate.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]

        # Input layernorm — fused (TE spec) or separate (local spec)
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "input_layernorm.weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]

        # Post-attention layernorm — fused (TE spec) or separate (local spec)
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # QK normalization (Qwen3)
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown Megatron parameter name: {name}")


# ============================================================================
# TP all-gather for weight sync
# ============================================================================

def _remap_layer_index(name: str, offset: int) -> str:
    """Shift local Megatron decoder layer indices to global indices.

    With PP>1, Megatron names layers 0..layers_per_stage-1 locally.
    HF checkpoint (and ``_convert_qwen_to_hf``) expects global indices.
    """
    def _add_offset(m):
        local_idx = int(m.group(1))
        return f"decoder.layers.{local_idx + offset}"

    return re.sub(r"decoder\.layers\.(\d+)", _add_offset, name)


def _remap_hf_layer_index(hf_name: str, offset: int) -> str:
    """Shift HF layer indices from PP-local to global.

    Bridge export produces HF-format names with local layer indices
    (e.g., ``model.layers.0`` on every PP stage).  This adds *offset*
    so that pp_rank=1's ``model.layers.0`` becomes ``model.layers.18``
    (for a 36-layer model with PP=2).

    Non-layer keys (``model.embed_tokens``, ``lm_head``, ``model.norm``)
    pass through unchanged.
    """
    def _add_offset(m):
        return f"model.layers.{int(m.group(1)) + offset}"

    return re.sub(r"model\.layers\.(\d+)", _add_offset, hf_name)


def _remap_expert_index(name: str, ep_rank: int, num_local_experts: int) -> str:
    """Shift local expert indices to global indices for EP>1.

    With EP>1, Megatron names local experts 0..num_local_experts-1.
    This remaps to global indices: ep_rank * num_local_experts + local_idx.
    Works on Megatron-format names (``decoder.layers.X.mlp.experts.local_experts.Y``).
    """
    def _add_offset(m):
        local_idx = int(m.group(1))
        global_idx = ep_rank * num_local_experts + local_idx
        return f"local_experts.{global_idx}"

    return re.sub(r"local_experts\.(\d+)", _add_offset, name)


def _remap_hf_expert_index(hf_name: str, ep_rank: int, num_local_experts: int) -> str:
    """Shift HF expert indices from EP-local to global.

    Works on HF-format names (``model.layers.X.mlp.experts.Y``).
    """
    def _add_offset(m):
        local_idx = int(m.group(1))
        global_idx = ep_rank * num_local_experts + local_idx
        return f"mlp.experts.{global_idx}"

    return re.sub(r"mlp\.experts\.(\d+)", _add_offset, hf_name)


def _gather_ep_expert_params(
    hf_state_dict: dict[str, torch.Tensor],
    ep_rank: int,
    ep_size: int,
    ep_group,
    cpu_stage: bool,
) -> dict[str, torch.Tensor]:
    """Gather expert parameters from all EP ranks onto ep_rank=0.

    Non-expert parameters (attention, layernorm, router, shared experts) are
    replicated across EP ranks — only ep_rank=0's copies are kept.
    Expert parameters differ per EP rank; each rank sends its subset to ep_rank=0.

    Follows the same broadcast-based pattern as ``_gather_pp_state_dicts``.
    """
    device = torch.cuda.current_device()
    ep_global_ranks = dist.get_process_group_ranks(ep_group)

    def _is_per_expert_param(key: str) -> bool:
        """Check if an HF-format key is a per-expert parameter (not shared/router)."""
        # Matches patterns like "model.layers.X.mlp.experts.Y.{gate,up,down}_proj.weight"
        return bool(re.search(r"mlp\.experts\.\d+\.", key))

    if ep_rank == 0:
        # Start with our own state dict (contains ep_rank=0's experts + all non-expert params)
        merged: dict[str, torch.Tensor] = {}
        for k, v in hf_state_dict.items():
            if cpu_stage and isinstance(v, torch.Tensor) and v.is_cuda:
                merged[k] = v.detach().cpu()
            else:
                merged[k] = v
        hf_state_dict.clear()
    else:
        merged = {}

    # Each non-zero EP rank broadcasts its expert parameters to ep_rank=0.
    for src_ep_rank in range(1, ep_size):
        # Phase 1: Broadcast metadata (expert param names/shapes/dtypes).
        if ep_rank == src_ep_rank:
            expert_items = {k: v for k, v in hf_state_dict.items() if _is_per_expert_param(k)}
            metadata = [
                (key, tuple(t.shape), str(t.dtype).replace("torch.", ""))
                for key, t in expert_items.items()
            ]
        else:
            metadata = None
            expert_items = {}
        metadata_list = [metadata]
        dist.broadcast_object_list(
            metadata_list,
            src=ep_global_ranks[src_ep_rank],
            group=ep_group,
        )
        metadata = metadata_list[0]

        # Phase 2: Broadcast each expert tensor via NCCL.
        for key, shape, dtype_name in metadata:
            dtype = getattr(torch, dtype_name)

            if ep_rank == src_ep_rank:
                tensor = expert_items[key]
                buf = tensor.to(device).contiguous() if not tensor.is_cuda else tensor.contiguous()
            else:
                buf = torch.empty(shape, dtype=dtype, device=device)

            dist.broadcast(buf, src=ep_global_ranks[src_ep_rank], group=ep_group)

            if ep_rank == 0:
                merged[key] = buf.cpu() if cpu_stage else buf
            del buf

        if ep_rank == src_ep_rank:
            hf_state_dict.clear()

    return merged if ep_rank == 0 else {}


def _gather_pp_state_dicts(
    local_state_dict: dict[str, torch.Tensor],
    pp_rank: int,
    pp_size: int,
    mpu,
    cpu_stage: bool,
) -> dict[str, torch.Tensor]:
    """Gather HF state dicts from all PP stages onto pp_rank=0.

    Uses tensor-level ``dist.broadcast`` within the PP group for each
    parameter.  Metadata (tensor names, shapes, dtypes) is sent once per
    source stage via ``dist.broadcast_object_list`` (small pickle, fast).

    Only tp_rank=0 ranks should call this (others have empty dicts).
    """
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_global_ranks = dist.get_process_group_ranks(pp_group)
    device = torch.cuda.current_device()

    # pp_rank=0 starts with its own state dict as the merged result.
    if pp_rank == 0:
        merged: dict[str, torch.Tensor] = {}
        for k, v in local_state_dict.items():
            if cpu_stage and isinstance(v, torch.Tensor) and v.is_cuda:
                merged[k] = v.detach().cpu()
            else:
                merged[k] = v
        local_state_dict.clear()
    else:
        merged = {}

    # Each non-zero PP stage broadcasts its tensors to pp_rank=0.
    for src_pp_rank in range(1, pp_size):
        # Phase 1: Broadcast metadata (small pickle — tensor names/shapes/dtypes).
        if pp_rank == src_pp_rank:
            metadata = [
                (key, tuple(t.shape), str(t.dtype).replace("torch.", ""))
                for key, t in local_state_dict.items()
            ]
        else:
            metadata = None
        metadata_list = [metadata]
        dist.broadcast_object_list(
            metadata_list,
            src=pp_global_ranks[src_pp_rank],
            group=pp_group,
        )
        metadata = metadata_list[0]

        # Phase 2: Broadcast each tensor via NCCL (no pickle).
        for key, shape, dtype_name in metadata:
            dtype = getattr(torch, dtype_name)

            if pp_rank == src_pp_rank:
                tensor = local_state_dict[key]
                buf = tensor.to(device).contiguous() if not tensor.is_cuda else tensor.contiguous()
            else:
                buf = torch.empty(shape, dtype=dtype, device=device)

            dist.broadcast(buf, src=pp_global_ranks[src_pp_rank], group=pp_group)

            if pp_rank == 0:
                merged[key] = buf.cpu() if cpu_stage else buf
            del buf

        if pp_rank == src_pp_rank:
            local_state_dict.clear()

    return merged if pp_rank == 0 else {}


def _all_gather_tp_param(name: str, param: torch.nn.Parameter, tp_group) -> torch.Tensor:
    """
    All-gather a TP-sharded parameter to reconstruct the full tensor.
    
    Handles:
    - Non-TP params (just return as-is)
    - Column-parallel (partition_dim=0): concat along dim 0
    - Row-parallel (partition_dim=1): concat along dim 1
    - GLU (gate+up fused in linear_fc1): re-interleave after gather
    """
    if not hasattr(param, "tensor_model_parallel") or not param.tensor_model_parallel:
        return param.data
    
    if getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    tp_size = dist.get_world_size(tp_group)
    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    
    partition_dim = param.partition_dim
    
    # GLU (gate+up fused): Megatron interleaves gate/up across TP ranks
    # Need to re-chunk and concatenate properly
    if "linear_fc1.weight" in name:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]

    # MoE linear_fc2 bug workaround
    if "linear_fc2.weight" in name and partition_dim == 0:
        partition_dim = 1

    return torch.cat(param_partitions, dim=partition_dim)


# ============================================================================
# Grad buffer memory management (pattern from SkyRL/verl)
# ============================================================================

def _free_grad_buffers(model_list: list) -> list[list[int]]:
    """Temporarily free Megatron DDP grad buffer GPU storage.

    After ``optimizer.step()`` the grad buffers are zeroed and unused until the
    next backward pass.  Releasing their backing storage reclaims ~14 GB for a
    14B model, giving headroom for weight broadcast.

    Returns the original storage sizes so ``_restore_grad_buffers`` can
    reallocate them.
    """
    sizes: list[list[int]] = []
    for model_chunk in model_list:
        chunk_sizes: list[int] = []
        all_buffers = list(model_chunk.buffers)
        # MoE models have separate expert parallel grad buffers
        if hasattr(model_chunk, "expert_parallel_buffers"):
            all_buffers.extend(model_chunk.expert_parallel_buffers)
        for buffer in all_buffers:
            sz = buffer.grad_data.storage().size()
            chunk_sizes.append(sz)
            if sz > 0:
                buffer.grad_data.storage().resize_(0)
        sizes.append(chunk_sizes)
    return sizes


def _restore_grad_buffers(model_list: list, sizes: list[list[int]]) -> None:
    """Restore grad buffer GPU storage freed by ``_free_grad_buffers``."""
    for model_chunk, chunk_sizes in zip(model_list, sizes):
        all_buffers = list(model_chunk.buffers)
        if hasattr(model_chunk, "expert_parallel_buffers"):
            all_buffers.extend(model_chunk.expert_parallel_buffers)
        for buffer, sz in zip(all_buffers, chunk_sizes):
            if sz > 0 and buffer.grad_data.storage().size() == 0:
                buffer.grad_data.storage().resize_(sz)
                buffer.grad_data.zero_()


# ============================================================================
# Model building
# ============================================================================

def _build_gpt_model(
    tf_config: "TransformerConfig",
    hf_config,
    padded_vocab_size: int,
    grad_reduce_in_fp32: bool = True,
) -> list:
    """
    Build GPTModel + Float16Module + DDP wrapper using only megatron.core APIs.
    """
    from megatron.core import parallel_state as mpu, tensor_parallel
    from megatron.core.distributed import (
        DistributedDataParallel as DDP,
        DistributedDataParallelConfig,
    )
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.module import Float16Module

    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()

    # Layer spec: TransformerEngine (fused attention, FP8) or local (no TE dependency)
    # Context parallelism requires TEDotProductAttention, so force TE when CP > 1.
    # For MoE models, num_experts is read from TransformerConfig (populated from HF config).
    num_experts = tf_config.num_moe_experts  # None for dense models, int for MoE
    use_te = bool(config.cfg.megatron_use_transformer_engine)
    if tf_config.context_parallel_size > 1 and not use_te:
        _log.info(
            "Context parallelism (CP=%d) requires TransformerEngine; enabling automatically",
            tf_config.context_parallel_size,
        )
        use_te = True
    if use_te:
        try:
            from megatron.core.models.gpt.gpt_layer_specs import (
                get_gpt_layer_with_transformer_engine_spec,
            )
            layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_experts,
                moe_grouped_gemm=False,
                qk_layernorm=tf_config.qk_layernorm,
            )
            _log.info("Using TransformerEngine layer spec%s",
                       f" (MoE: {num_experts} experts)" if num_experts else "")
        except (ImportError, NameError):
            if tf_config.context_parallel_size > 1:
                raise ImportError(
                    "Context parallelism (CP > 1) requires TransformerEngine but it is not "
                    "installed. Install transformer-engine or set megatron_context_parallel_size=1."
                )
            _log.warning(
                "megatron_use_transformer_engine=true but transformer-engine not installed, "
                "falling back to local spec"
            )
            use_te = False

    if not use_te:
        layer_spec = get_gpt_layer_local_spec(
            num_experts=num_experts,
            moe_grouped_gemm=False,
            qk_layernorm=tf_config.qk_layernorm,
            normalization=tf_config.normalization,
        )

    # Build GPTModel
    model = GPTModel(
        config=tf_config,
        transformer_layer_spec=layer_spec,
        vocab_size=padded_vocab_size,
        max_sequence_length=getattr(hf_config, "max_position_embeddings", 32768),
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=getattr(
            hf_config, "tie_word_embeddings", False
        ),
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=int(getattr(hf_config, "rope_theta", 10000)),
    )

    # Set default TP attributes on all params
    for param in model.parameters():
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Move to GPU
    model.cuda(torch.cuda.current_device())

    # Float16 wrapping (bf16/fp16)
    if tf_config.bf16 or tf_config.fp16:
        model = Float16Module(tf_config, model)

    # DDP wrapping
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        overlap_grad_reduce=bool(config.cfg.megatron_overlap_grad_reduce),
        use_distributed_optimizer=bool(config.cfg.megatron_use_distributed_optimizer),
    )
    model = DDP(
        config=tf_config,
        ddp_config=ddp_config,
        module=model,
        disable_bucketing=False,
    )
    model.broadcast_params()

    return [model]


# ============================================================================
# HF checkpoint loading into Megatron model
# ============================================================================

def _create_bridge(model_path: str):
    """Try to create a megatron.bridge instance.

    Returns the bridge object or ``None`` if the library is not available.
    """
    try:
        from megatron.bridge import AutoBridge
        return AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
    except ImportError:
        pass
    except Exception as e:
        _log.info(f"megatron.bridge init failed ({e})")

    return None


def _load_hf_checkpoint(model_list: list, model_path: str, rank: int,
                         bridge=None) -> None:
    """
    Load HuggingFace checkpoint into a Megatron model.

    Uses the provided *bridge* instance when available.
    Falls back to manual safetensors loading with TP slicing.
    """
    if bridge is not None:
        try:
            _log.info(f"Loading HF checkpoint via bridge: {model_path}", rank=rank)
            # megatron.bridge uses load_hf_weights(model_list)
            try:
                bridge.load_hf_weights(model_list, model_path)
            except TypeError:
                bridge.load_hf_weights(model_list)
            _log.info("HF checkpoint loaded via bridge", rank=rank)
            return
        except Exception as e:
            _log.info(f"Bridge loading failed ({e}), falling back to manual loading", rank=rank)

    # Fallback: manual safetensors loading with TP slicing
    _load_hf_checkpoint_manual(model_list, model_path, rank)


def _resolve_model_path(model_path: str) -> str:
    """
    Resolve a model path to a local directory.
    
    If model_path is already a local directory, return it as-is.
    If it's a HuggingFace Hub ID (e.g., "Qwen/Qwen2.5-3B"), download/resolve
    it to the local HuggingFace cache.
    """
    if os.path.isdir(model_path):
        return model_path
    
    # It's a HuggingFace Hub ID — resolve to local cache
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(model_path)
        _log.info(f"Resolved HF Hub path '{model_path}' -> '{local_path}'")
        return local_path
    except ImportError:
        _log.warning(
            "huggingface_hub is not installed; cannot download '%s'. "
            "Trying transformers cached_file fallback.",
            model_path,
        )
    except Exception as exc:
        _log.warning(
            "snapshot_download('%s') failed: %s. "
            "Trying transformers cached_file fallback.",
            model_path, exc,
        )

    # Fallback: try transformers' cached path resolution
    try:
        from transformers.utils import cached_file
        # Get path to any file in the repo to find the cache dir
        resolved = cached_file(model_path, "config.json")
        if resolved:
            return os.path.dirname(resolved)
    except ImportError:
        _log.warning(
            "transformers is not installed; cannot resolve '%s' via cached_file.",
            model_path,
        )
    except Exception as exc:
        _log.warning(
            "cached_file('%s', 'config.json') failed: %s",
            model_path, exc,
        )

    # Last resort: return as-is (will fail with clear error on file open)
    _log.warning(
        "Could not resolve model path '%s' via huggingface_hub or transformers; "
        "using raw path (will fail if not a local directory).",
        model_path,
    )
    return model_path


def _load_hf_checkpoint_manual(model_list: list, model_path: str, rank: int) -> None:
    """
    Manual HF checkpoint loading with TP shard slicing.
    
    This handles the HF → Megatron weight name mapping and TP sharding.
    It reads safetensors files and loads the appropriate shard for this rank.
    
    """
    import json
    from megatron.core import parallel_state as mpu
    from safetensors import safe_open
    from transformers import AutoConfig

    # Resolve HF Hub ID to local path
    local_model_path = _resolve_model_path(model_path)

    hf_config = AutoConfig.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    # Build a name mapping: mcore_name -> list of (hf_safetensor_file, hf_key) pairs
    # First, read the safetensors index to know which file has which weights
    index_file = os.path.join(local_model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]  # hf_key -> filename
    else:
        # Single safetensors file
        weight_map = None

    # Collect all mcore parameter names and shapes
    mcore_params = {}
    for model_chunk in model_list:
        for name, param in model_chunk.named_parameters():
            mcore_params[name] = param

    # Build reverse name mapping (mcore_name -> list of hf_keys)
    # This is model-specific. For Qwen-family models:
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
    num_heads = hf_config.num_attention_heads
    hidden = hf_config.hidden_size
    head_dim = getattr(hf_config, "head_dim", hidden // num_heads)

    loaded_count = 0
    skipped_count = 0

    # Open all safetensors files
    safetensor_files = {}

    def get_hf_tensor(hf_key):
        """Load a tensor from the appropriate safetensors file."""
        if weight_map is not None:
            filename = weight_map.get(hf_key)
            if filename is None:
                return None
            filepath = os.path.join(local_model_path, filename)
        else:
            filepath = os.path.join(local_model_path, "model.safetensors")
        
        if filepath not in safetensor_files:
            safetensor_files[filepath] = safe_open(filepath, framework="pt", device="cpu")
        return safetensor_files[filepath].get_slice(hf_key)

    # Compute global layer offset for this PP stage.
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    num_layers = hf_config.num_hidden_layers
    layers_per_stage = num_layers // pp_size
    layer_offset = pp_rank * layers_per_stage

    # Compute expert offset for this EP rank (MoE models only).
    try:
        ep_rank = mpu.get_expert_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
    except Exception:
        ep_rank = 0
        ep_size = 1
    num_moe_experts = getattr(hf_config, "num_experts", None) or \
                      getattr(hf_config, "num_local_experts", None)
    if num_moe_experts and ep_size > 1:
        expert_offset = ep_rank * (num_moe_experts // ep_size)
    else:
        expert_offset = 0

    for mcore_name, param in mcore_params.items():
        try:
            _load_single_param(
                mcore_name, param, get_hf_tensor,
                hf_config, tp_rank, tp_size,
                layer_offset=layer_offset,
                expert_offset=expert_offset,
            )
            loaded_count += 1
        except Exception as e:
            _log.warning(f"Failed to load {mcore_name}: {e}", rank=rank)
            skipped_count += 1

    # Close safetensors files
    safetensor_files.clear()

    _log.info(
        f"Manual checkpoint loading: {loaded_count} params loaded, {skipped_count} skipped",
        rank=rank,
    )


def _load_single_param(mcore_name, param, get_hf_tensor, hf_config, tp_rank, tp_size,
                       layer_offset: int = 0, expert_offset: int = 0):
    """Load a single Megatron param from HF checkpoint with TP slicing.

    Args:
        layer_offset: Global layer index offset for this PP stage.
            With PP>1, Megatron uses local indices (0..layers_per_stage-1)
            but HF checkpoints use global indices (0..num_layers-1).
        expert_offset: Global expert index offset for this EP rank.
            With EP>1, Megatron uses local expert indices (0..num_local_experts-1)
            but HF checkpoints use global indices (0..num_experts-1).
    """
    # Parse mcore_name to figure out which HF keys we need
    # This is the reverse of _convert_qwen_to_hf
    hidden = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    head_dim = getattr(hf_config, "head_dim", hidden // num_heads)
    group_size = num_heads // num_kv_heads

    # Extract layer index if present, applying PP offset for global index.
    layer_match = re.search(r"decoder\.layers\.(\d+)", mcore_name)
    layer_idx = str(int(layer_match.group(1)) + layer_offset) if layer_match else None

    def tp_slice(tensor, dim):
        """Take TP shard from a full tensor."""
        size = tensor.shape[dim]
        per_tp = size // tp_size
        start = tp_rank * per_tp
        slices = [slice(None)] * len(tensor.shape)
        slices[dim] = slice(start, start + per_tp)
        return tensor[tuple(slices)].contiguous()

    with torch.no_grad():
        # Embedding
        if "embedding.word_embeddings.weight" in mcore_name:
            t = get_hf_tensor("model.embed_tokens.weight")
            full = t[:]
            # Pad vocab if needed, then TP slice
            padded_size = param.shape[0] * tp_size
            if full.shape[0] < padded_size:
                full = F.pad(full, (0, 0, 0, padded_size - full.shape[0]))
            param.data.copy_(tp_slice(full, 0))
            return

        # Output layer
        if "output_layer.weight" in mcore_name:
            t = get_hf_tensor("lm_head.weight")
            if t is None:
                # tie_word_embeddings=True: lm_head.weight not stored separately,
                # share from embedding layer.
                t = get_hf_tensor("model.embed_tokens.weight")
            full = t[:]
            padded_size = param.shape[0] * tp_size
            if full.shape[0] < padded_size:
                full = F.pad(full, (0, 0, 0, padded_size - full.shape[0]))
            param.data.copy_(tp_slice(full, 0))
            return

        # Final layernorm
        if "decoder.final_layernorm.weight" in mcore_name:
            t = get_hf_tensor("model.norm.weight")
            param.data.copy_(t[:])
            return

        if layer_idx is None:
            raise ValueError(f"Cannot map mcore param: {mcore_name}")

        # QKV fused weight
        if "self_attention.linear_qkv.weight" in mcore_name:
            q = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight")[:]
            k = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight")[:]
            v = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.weight")[:]
            # Interleave into mcore format: [kv_heads, (group_size+2)*head_dim, hidden]
            q = q.view(num_kv_heads, group_size, head_dim, hidden)
            k = k.view(num_kv_heads, 1, head_dim, hidden)
            v = v.view(num_kv_heads, 1, head_dim, hidden)
            fused = torch.cat([q, k, v], dim=1).reshape(-1, hidden)
            param.data.copy_(tp_slice(fused, 0))
            return

        # QKV fused bias
        if "self_attention.linear_qkv.bias" in mcore_name:
            q = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.bias")[:]
            k = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.bias")[:]
            v = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.bias")[:]
            q = q.view(num_kv_heads, group_size * head_dim)
            k = k.view(num_kv_heads, head_dim)
            v = v.view(num_kv_heads, head_dim)
            fused = torch.cat([q, k, v], dim=1).flatten()
            param.data.copy_(tp_slice(fused.unsqueeze(0), 1).squeeze(0) if tp_size > 1 else fused)
            return

        # Output projection
        if "self_attention.linear_proj.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.weight")[:]
            param.data.copy_(tp_slice(t, 1))
            return

        # Gate + Up fused (linear_fc1)
        if "mlp.linear_fc1.weight" in mcore_name:
            gate = get_hf_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight")[:]
            up = get_hf_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight")[:]
            gate = tp_slice(gate, 0)
            up = tp_slice(up, 0)
            param.data.copy_(torch.cat([gate, up], dim=0))
            return

        # Down projection (linear_fc2)
        if "mlp.linear_fc2.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight")[:]
            param.data.copy_(tp_slice(t, 1))
            return

        # MoE router weight (not TP-sharded)
        if "mlp.router.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.mlp.gate.weight")
            param.data.copy_(t[:])
            return

        # MoE expert FC1 (gate+up fused)
        expert_fc1 = re.search(
            r"mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight", mcore_name
        )
        if expert_fc1:
            global_idx = int(expert_fc1.group(1)) + expert_offset
            gate = get_hf_tensor(f"model.layers.{layer_idx}.mlp.experts.{global_idx}.gate_proj.weight")[:]
            up = get_hf_tensor(f"model.layers.{layer_idx}.mlp.experts.{global_idx}.up_proj.weight")[:]
            param.data.copy_(torch.cat([gate, up], dim=0))
            return

        # MoE expert FC2 (down projection)
        expert_fc2 = re.search(
            r"mlp\.experts\.local_experts\.(\d+)\.linear_fc2\.weight", mcore_name
        )
        if expert_fc2:
            global_idx = int(expert_fc2.group(1)) + expert_offset
            t = get_hf_tensor(f"model.layers.{layer_idx}.mlp.experts.{global_idx}.down_proj.weight")[:]
            param.data.copy_(t)
            return

        # MoE shared expert FC1 (gate+up fused)
        if "mlp.shared_experts.linear_fc1.weight" in mcore_name:
            gate = get_hf_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight")[:]
            up = get_hf_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight")[:]
            param.data.copy_(torch.cat([gate, up], dim=0))
            return

        # MoE shared expert FC2
        if "mlp.shared_experts.linear_fc2.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight")[:]
            param.data.copy_(t)
            return

        # MoE shared expert gate
        if "mlp.shared_experts.gate.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight")[:]
            param.data.copy_(t)
            return

        # Input layernorm — fused into QKV (TE spec)
        if "self_attention.linear_qkv.layer_norm_weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.input_layernorm.weight")[:]
            param.data.copy_(t)
            return

        # Input layernorm — separate module (local spec)
        if "input_layernorm.weight" in mcore_name and "self_attention" not in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.input_layernorm.weight")[:]
            param.data.copy_(t)
            return

        # Post-attention layernorm — fused into FC1 (TE spec)
        if "mlp.linear_fc1.layer_norm_weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight")[:]
            param.data.copy_(t)
            return

        # Post-attention layernorm — separate module (local spec)
        if "pre_mlp_layernorm.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight")[:]
            param.data.copy_(t)
            return

        # QK normalization (Qwen3)
        if "self_attention.q_layernorm.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.q_norm.weight")[:]
            param.data.copy_(t)
            return
        if "self_attention.k_layernorm.weight" in mcore_name:
            t = get_hf_tensor(f"model.layers.{layer_idx}.self_attn.k_norm.weight")[:]
            param.data.copy_(t)
            return

    raise ValueError(f"Cannot map mcore param: {mcore_name}")


# ============================================================================
# MegatronBackend
# ============================================================================

class MegatronBackend(TrainingBackend):
    """
    Megatron-Core training backend.
    
    Supports tensor parallelism (TP), pipeline parallelism (PP),
    context parallelism (CP), and expert parallelism (EP) for MoE models
    via NVIDIA Megatron-Core.
    
    Only depends on megatron.core (uv add megatron-core megatron-bridge).
    Does NOT require the full Megatron-LM repository.

    Requirements:
        - megatron-core >= 0.9.0 (uv add megatron-core megatron-bridge)
        - CUDA_DEVICE_MAX_CONNECTIONS=1 (set by ray_runtime)
    """

    def __init__(self):
        self._rank = 0
        self._local_rank = 0
        self._world_size = 1
        self._device = torch.device("cpu")
        self._model = None  # List of DDP-wrapped model chunks
        self._optimizer = None
        self._scheduler = None
        self._tf_config = None  # Megatron TransformerConfig
        self._hf_config = None  # HuggingFace model config
        self._tokenizer = None
        self._padded_vocab_size = 0
        self._bridge = None  # megatron.bridge instance for weight conversion

    def init(self) -> dict:
        """
        Initialize Megatron: process groups, model, optimizer, checkpoint.
        
        Flow:
        1. Initialize torch.distributed
        2. Initialize model parallel groups (TP, PP, CP, EP, DP) via megatron.core
        3. Build TransformerConfig directly from HF config
        4. Build GPTModel + DDP wrapper + optimizer
        5. Load HF checkpoint weights
        """
        setup_logging()
        if _maybe_install_unified_memory_stub():
            _log.info("Installed Megatron unified-memory stub (startup JIT bypass)")

        from megatron.core import parallel_state as mpu, tensor_parallel
        
        # Read distributed info from torchrun environment
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._device = torch.device(f"cuda:{self._local_rank}")
        torch.cuda.set_device(self._local_rank)
        
        _log.banner("Megatron Backend Init")

        tp_size = config.cfg.megatron_tensor_parallel_size
        pp_size = config.cfg.megatron_pipeline_parallel_size
        cp_size = config.cfg.megatron_context_parallel_size
        ep_size = config.cfg.megatron_expert_parallel_size
        
        _log.info(
            f"world_size={self._world_size}, rank={self._rank}, "
            f"TP={tp_size}, PP={pp_size}, CP={cp_size}, EP={ep_size}",
            rank=self._rank,
        )

        # Sequence parallel: only meaningful when TP > 1.
        enable_sequence_parallel = bool(config.cfg.megatron_sequence_parallel)
        if enable_sequence_parallel and tp_size > 1 and not _can_use_sequence_parallel():
            enable_sequence_parallel = False
            _log.info(
                "Sequence parallel requires TransformerEngine (or Apex with LayerNorm); "
                "disabling for RMSNorm with local spec",
                rank=self._rank,
            )
        
        # Step 1: Initialize torch.distributed (torchrun sets up env vars)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Step 2: Initialize megatron model parallel groups
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
        )
        
        # Random seeds (deterministic across TP/PP ranks)
        seed = 42 + 100 * mpu.get_pipeline_model_parallel_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
        
        _log.info("Megatron model parallel groups initialized", rank=self._rank)
        
        # Step 3: Load HF config and tokenizer in parallel, then build TransformerConfig
        from transformers import AutoConfig, AutoTokenizer
        with ThreadPoolExecutor(max_workers=2) as pool:
            cfg_future = pool.submit(AutoConfig.from_pretrained, config.cfg.model, trust_remote_code=True)
            tok_future = pool.submit(AutoTokenizer.from_pretrained, config.cfg.model, trust_remote_code=True)
            self._hf_config = cfg_future.result()
            self._tokenizer = tok_future.result()

        # Validate EP configuration against the model
        num_moe_experts = getattr(self._hf_config, "num_experts", None) or \
                          getattr(self._hf_config, "num_local_experts", None)
        if ep_size > 1 and num_moe_experts is None:
            raise ValueError(
                f"megatron_expert_parallel_size={ep_size} but model "
                f"'{config.cfg.model}' has no MoE experts. "
                f"Set megatron_expert_parallel_size=1 for dense models."
            )
        if num_moe_experts is not None and ep_size > 1 and num_moe_experts % ep_size != 0:
            raise ValueError(
                f"num_moe_experts ({num_moe_experts}) must be divisible by "
                f"megatron_expert_parallel_size ({ep_size})"
            )

        self._tf_config = _hf_to_transformer_config(
            hf_config=self._hf_config,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            ep_size=ep_size,
            dtype=torch.bfloat16,
            seq_parallel=enable_sequence_parallel,
            gradient_checkpointing=bool(config.cfg.megatron_gradient_checkpointing),
        )
        
        # Step 4: Compute padded vocab size
        self._padded_vocab_size = _pad_vocab_size(
            self._hf_config.vocab_size, tp_size
        )
        
        # Step 5: Create bridge for weight conversion (megatron.bridge or None)
        self._bridge = _create_bridge(config.cfg.model)
        if self._bridge is not None:
            _log.info("Weight conversion bridge available", rank=self._rank)

        # Step 6: Build model + optimizer + load checkpoint
        self._build_model_and_optimizer()
        self._scheduler = build_lr_scheduler(self._optimizer)

        _log.info(
            f"dp_rank={self.dp_rank}, dp_world_size={self.dp_world_size}",
            rank=self._rank,
        )
        
        # Collect parallelism ranks for the setup snapshot.
        tp_rank = int(mpu.get_tensor_model_parallel_rank())
        pp_rank = int(mpu.get_pipeline_model_parallel_rank())
        try:
            cp_rank = int(mpu.get_context_parallel_rank())
        except Exception:
            cp_rank = 0
        try:
            ep_rank = int(mpu.get_expert_model_parallel_rank())
        except Exception:
            ep_rank = 0

        return {
            "rank": self._rank,
            "local_rank": self._local_rank,
            "world_size": self._world_size,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "device": self._device,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "cp_rank": cp_rank,
            "cp_size": cp_size,
            "ep_rank": ep_rank,
            "ep_size": ep_size,
        }
    
    def _build_model_and_optimizer(self):
        """
        Build Megatron GPTModel, load checkpoint, then create optimizer.

        Uses only megatron.core APIs. No megatron.training dependency.
        """
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

        # Build model
        self._model = _build_gpt_model(
            tf_config=self._tf_config,
            hf_config=self._hf_config,
            padded_vocab_size=self._padded_vocab_size,
            grad_reduce_in_fp32=bool(config.cfg.megatron_grad_reduce_in_fp32),
        )
        
        # Prepare optimizer config
        optimizer_cpu_offload = bool(config.cfg.megatron_optimizer_cpu_offload)
        optimizer_offload_fraction = float(
            getattr(
                config.cfg,
                "megatron_optimizer_offload_fraction",
                1.0 if optimizer_cpu_offload else 0.0,
            )
        )
        overlap_cpu_optimizer_d2h_h2d = bool(config.cfg.megatron_overlap_cpu_optimizer_d2h_h2d)
        use_precision_aware_optimizer = bool(config.cfg.megatron_use_precision_aware_optimizer)
        main_grads_dtype = _torch_dtype_from_config(
            config.cfg.megatron_main_grads_dtype,
            torch.float32,
        )
        main_params_dtype = _torch_dtype_from_config(
            config.cfg.megatron_main_params_dtype,
            torch.float32,
        )
        exp_avg_dtype = _torch_dtype_from_config(
            config.cfg.megatron_exp_avg_dtype,
            torch.float32,
        )
        exp_avg_sq_dtype = _torch_dtype_from_config(
            config.cfg.megatron_exp_avg_sq_dtype,
            torch.float32,
        )

        if optimizer_cpu_offload:
            _log.info(
                "Megatron optimizer CPU offload enabled "
                f"(fraction={optimizer_offload_fraction}, "
                f"overlap_d2h_h2d={overlap_cpu_optimizer_d2h_h2d})",
                rank=self._rank,
            )

        opt_config = OptimizerConfig(
            optimizer="adam",
            lr=config.cfg.learning_rate,
            weight_decay=config.cfg.weight_decay,
            bf16=True,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_eps=1e-8,
            clip_grad=config.cfg.grad_clip,
            use_distributed_optimizer=bool(config.cfg.megatron_use_distributed_optimizer),
            params_dtype=torch.bfloat16,
            optimizer_cpu_offload=optimizer_cpu_offload,
            optimizer_offload_fraction=optimizer_offload_fraction,
            overlap_cpu_optimizer_d2h_h2d=overlap_cpu_optimizer_d2h_h2d,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
            main_grads_dtype=main_grads_dtype,
            main_params_dtype=main_params_dtype,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
        )

        # Load checkpoint (HF format) before optimizer creation.
        # This is especially important for CPU-offloaded optimizer states, which
        # keep CPU mirrors of model params at optimizer init time.
        _load_hf_checkpoint(self._model, config.cfg.model, self._rank,
                            bridge=self._bridge)

        self._optimizer = get_megatron_optimizer(
            opt_config,
            self._model,
        )
        
        torch.cuda.empty_cache()
        
        # Log model size
        total_params = sum(
            p.numel() for model_chunk in self._model
            for p in model_chunk.parameters()
        )
        _log.info(
            f"Model: {total_params / 1e6:.2f}M params, "
            f"padded_vocab={self._padded_vocab_size} (orig={self._hf_config.vocab_size})",
            rank=self._rank,
        )

    # ================================================================
    # Training
    # ================================================================

    def _compute_batch_logprobs(self, micro_batches: list[dict]) -> None:
        """Compute logprobs with the current policy for all micro-batches (no grad).

        TP-aware: uses ``_vocab_parallel_log_probs_and_entropy`` when tp > 1.
        PP-aware: uses Megatron's pipeline schedule with ``forward_only=True``
        and broadcasts logprobs from the last PP stage to all stages.

        Stores ``batch_logprobs`` in each micro-batch dict, in the same shape as
        ``vllm_logprobs`` (``[batch, seq_len]`` with 0.0 at position 0).
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.utils import get_model_config

        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group() if tp_world_size > 1 else None
        pp_size = mpu.get_pipeline_model_parallel_world_size()

        for model_chunk in self._model:
            model_chunk.eval()

        # Collected logprobs per micro-batch (populated on last PP stage).
        collected_logprobs: list[torch.Tensor] = []
        # megatron_data is populated below; the closure reads chunked input_ids
        # from it so that shapes match the (possibly CP-chunked) logits.
        megatron_data: list[dict] = []

        def _logprob_loss_func(logits):
            """Loss func for forward-only logprob computation.

            Called only on the last PP stage by Megatron's schedule.
            Computes logprobs from logits and stashes them in collected_logprobs.
            Returns a dummy loss (required by Megatron's 3-tuple contract).
            """
            nonlocal collected_logprobs

            # Use the *prepared* (possibly CP-chunked) input_ids so that the
            # sequence dimension matches the logits produced by the model.
            mb_idx = len(collected_logprobs)
            input_ids = megatron_data[mb_idx]["input_ids"]
            orig_len = input_ids.shape[-1]

            if logits.shape[1] > orig_len:
                logits = logits[:, :orig_len, :]

            shift_logits = logits[..., :-1, :] / config.cfg.get_sampling_params()["temperature"]
            labels = input_ids[:, 1:]

            if tp_world_size > 1:
                log_probs, _ = _vocab_parallel_log_probs_and_entropy(
                    shift_logits, labels, tp_group,
                )
            else:
                lp_full = shift_logits.float().log_softmax(dim=-1)
                log_probs = torch.gather(lp_full, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            # Pad to [batch, local_seq_len] with 0.0 at position 0.
            batch_logprobs = torch.zeros(
                input_ids.shape[0], input_ids.shape[1],
                device=log_probs.device, dtype=log_probs.dtype,
            )
            batch_logprobs[..., 1:] = log_probs
            collected_logprobs.append(batch_logprobs)

            # Megatron expects (loss, num_tokens, loss_reduced_dict).
            dummy_loss = torch.tensor(0.0, device=logits.device, requires_grad=False)
            num_tokens = torch.tensor(1, dtype=torch.int64, device=logits.device)
            return dummy_loss, num_tokens, {"keys": [], "values": torch.tensor([], device=logits.device)}

        with torch.no_grad():
            # Prepare data for the pipeline schedule (applies CP chunking).
            megatron_data[:] = self._prepare_megatron_data(micro_batches)
            data_iter = iter(megatron_data)

            def forward_step_logprobs(data_iterator, model):
                batch = next(data_iterator)
                output_tensor = model(
                    input_ids=batch["tokens"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=None,
                )
                return output_tensor, _logprob_loss_func

            model_config = get_model_config(self._model[0])
            model_config.grad_scale_func = None
            model_config.timers = None

            # With CP, each rank processes seq_len/cp_size tokens.
            cp_ws = mpu.get_context_parallel_world_size()
            local_seq_len = config.cfg.seq_len // max(cp_ws, 1)

            forward_backward_func = get_forward_backward_func()
            forward_backward_func(
                forward_step_func=forward_step_logprobs,
                data_iterator=[data_iter],
                model=self._model,
                num_microbatches=len(micro_batches),
                seq_length=local_seq_len,
                micro_batch_size=1,
                forward_only=True,
            )

        # With PP > 1, logprobs are only on the last stage.  Broadcast to all.
        is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)
        cp_ws = mpu.get_context_parallel_world_size()

        for mb_idx, mb in enumerate(micro_batches):
            # Use the local (possibly CP-chunked) seq_len from prepared data.
            local_input_ids = megatron_data[mb_idx]["input_ids"]
            local_seq_len = local_input_ids.shape[-1]

            if is_last_stage:
                batch_logprobs = collected_logprobs[mb_idx]
            else:
                batch_logprobs = torch.zeros(
                    local_input_ids.shape[0], local_seq_len,
                    device=self._device, dtype=torch.float32,
                )

            if pp_size > 1:
                dist.broadcast(
                    batch_logprobs,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                )

            mb["batch_logprobs"] = batch_logprobs
            # Mark as already CP-chunked so _prepare_megatron_data won't
            # re-chunk during the subsequent training forward pass.
            if cp_ws > 1:
                mb["_batch_logprobs_is_cp_chunked"] = True

        for model_chunk in self._model:
            model_chunk.train()

    def train_step(
        self,
        trainer_data: dict,
        tracker: GPUTimelineLogger | _NullTracker | None = None,
    ) -> dict:
        """
        Execute one RL training step using Megatron's forward/backward schedule.

        When ``NUMBER_OF_MINIBATCHES > 1``, micro-batches are split into that many
        groups, each getting its own zero_grad → forward/backward → optimizer.step()
        cycle.

        Timeline events emitted (when tracker provided):
        - zero_grad
        - prepare_data
        - forward (per microbatch)
        - prepare_tensors (per microbatch)
        - loss_computation (per microbatch)
        - compute_entropy (per microbatch)
        - compute_kl (per microbatch)
        - backward (per microbatch)
        - finalize_model_grads
        - optimizer

        """
        def _track(name: str, **kwargs):
            if tracker is not None:
                return tracker.track(name, **kwargs)
            return nullcontext()

        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.utils import get_model_config
        from megatron.core.distributed import finalize_model_grads

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

            # Prepare data for this group
            with _track("prepare_data", minibatch=mb_idx):
                megatron_data = self._prepare_megatron_data(group)
                data_iter = iter(megatron_data)

            # Zero gradients
            with _track("zero_grad", minibatch=mb_idx):
                for model_chunk in self._model:
                    model_chunk.zero_grad_buffer()
                self._optimizer.zero_grad()

            # Train mode
            for model_chunk in self._model:
                model_chunk.train()

            # Setup training config
            model_config = get_model_config(self._model[0])
            model_config.grad_scale_func = self._optimizer.scale_loss
            model_config.timers = None

            # ------------------------------------------------------------------
            # Per-microbatch backward timing via CUDA-event "bookends".
            # ------------------------------------------------------------------
            _can_record_backward = (
                tracker is not None
                and hasattr(tracker, "_pending_cuda")
                and torch.cuda.is_available()
            )
            _bwd_state: dict = {"start": None, "micro_idx": -1}

            def _close_backward_event():
                """Record the end-event for the previous microbatch's backward."""
                if _can_record_backward and _bwd_state["start"] is not None:
                    end_ev = torch.cuda.Event(enable_timing=True)
                    end_ev.record()
                    tracker._pending_cuda.append((
                        "backward",
                        _bwd_state["start"],
                        end_ev,
                        _bwd_state["micro_idx"],
                        mb_idx,
                    ))
                    _bwd_state["start"] = None

            def _tracked_finalize_grads(models, *args, **kwargs):
                _close_backward_event()
                with _track("finalize_model_grads", minibatch=mb_idx):
                    return finalize_model_grads(models, *args, **kwargs)

            model_config.finalize_model_grads_func = _tracked_finalize_grads

            def forward_step(data_iterator, model):
                batch = next(data_iterator)
                tokens = batch["tokens"]
                micro_idx = int(batch.get("micro_idx", -1))

                _close_backward_event()

                with _track("forward", microbatch=micro_idx, minibatch=mb_idx):
                    output_tensor = model(
                        input_ids=tokens,
                        position_ids=batch.get("position_ids"),
                        attention_mask=batch.get("attention_mask"),
                        labels=None,
                    )

                return output_tensor, partial(
                    self._loss_func,
                    batch=batch,
                    num_micro_batches=num_mb_in_group,
                    tracker=tracker,
                    backward_cuda_state=_bwd_state if _can_record_backward else None,
                    minibatch=mb_idx,
                )

            # Run forward/backward via Megatron's pipeline schedule.
            # With CP, each rank processes seq_len/cp_size tokens.
            cp_ws = mpu.get_context_parallel_world_size()
            local_seq_len = config.cfg.seq_len // max(cp_ws, 1)

            forward_backward_func = get_forward_backward_func()
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=[data_iter],
                model=self._model,
                num_microbatches=num_mb_in_group,
                seq_length=local_seq_len,
                micro_batch_size=1,
                forward_only=False,
            )

            _close_backward_event()

            # Optimizer step
            with _track("optimizer", minibatch=mb_idx):
                _, grad_norm, _ = self._optimizer.step()

            # Release grads
            for model_chunk in self._model:
                model_chunk.zero_grad_buffer()
            self._optimizer.zero_grad()

            # Collect metrics for this group
            group_metrics = self._collect_metrics(losses_reduced, grad_norm, num_mb_in_group)

            # With PP > 1, entropy/KL are only valid on the last pipeline stage.
            # Broadcast them so the reporting rank (pp_rank=0) has correct values.
            from megatron.core import parallel_state as mpu
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            if pp_size > 1:
                metrics_tensor = torch.tensor(
                    [group_metrics.get("entropy", 0.0),
                     group_metrics.get("kl_divergence_inference", 0.0)],
                    device=self._device, dtype=torch.float32,
                )
                dist.broadcast(
                    metrics_tensor,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                )
                group_metrics["entropy"] = metrics_tensor[0].item()
                group_metrics["kl_divergence_inference"] = metrics_tensor[1].item()

            all_grad_norms.append(group_metrics.get("grad_norm", 0.0))

            # Token-weighted accumulation from per-group metrics.
            # _collect_metrics already averages across micro-batches, so we use
            # num_mb_in_group as a proxy weight (exact token counts are inside
            # _collect_metrics and not easily extractable).
            weighted_entropy += group_metrics.get("entropy", 0.0) * num_mb_in_group
            weighted_kl += group_metrics.get("kl_divergence_inference", 0.0) * num_mb_in_group
            total_tokens += num_mb_in_group

        # Step the LR scheduler once per train_step (NOT per minibatch group).
        if self._scheduler is not None:
            self._scheduler.step()

        # Aggregate metrics across all minibatch groups.
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

    def _prepare_megatron_data(self, micro_batches: list[dict]) -> list[dict]:
        """
        Convert our micro_batch format to Megatron's expected format.

        Our format: each micro_batch has input_ids [1, seq_len], loss_mask, advantages, etc.
        Megatron format: tokens [1, seq_len], plus optional segment attention mask.

        When context parallelism (CP) is active, all per-token tensors are split
        into contiguous chunks so each CP rank receives its ``seq_len / cp_size``
        portion of the sequence.  Position IDs keep their original values (needed
        for correct RoPE).
        """
        from megatron.core import parallel_state as mpu

        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank() if cp_size > 1 else 0

        def _cp_chunk(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
            """Contiguous-chunk a tensor along *dim* for the local CP rank."""
            return torch.chunk(t, cp_size, dim=dim)[cp_rank].contiguous()

        prepared = []
        for micro_idx, mb in enumerate(micro_batches):
            input_ids = mb["input_ids"].to(self._device)
            loss_mask = mb["loss_mask"].to(self._device)
            advantages = mb["advantages"].to(self._device)
            vllm_logprobs = mb.get("vllm_logprobs")
            if vllm_logprobs is not None:
                vllm_logprobs = vllm_logprobs.to(self._device)
            position_ids = mb.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(self._device)

            batch_logprobs = mb.get("batch_logprobs")
            if batch_logprobs is not None:
                batch_logprobs = batch_logprobs.to(self._device)

            # --- Context parallelism: chunk all per-token tensors ---
            if cp_size > 1:
                input_ids = _cp_chunk(input_ids)
                loss_mask = _cp_chunk(loss_mask)
                advantages = _cp_chunk(advantages)
                if vllm_logprobs is not None:
                    vllm_logprobs = _cp_chunk(vllm_logprobs)
                if position_ids is not None:
                    position_ids = _cp_chunk(position_ids)
                # batch_logprobs from _compute_batch_logprobs is already at
                # chunked length — skip re-chunking to avoid double-split.
                if batch_logprobs is not None and not mb.get("_batch_logprobs_is_cp_chunked"):
                    batch_logprobs = _cp_chunk(batch_logprobs)

            attention_mask = _build_packed_segment_attention_mask_from_position_ids(position_ids)

            tokens = input_ids

            prepared.append({
                "tokens": tokens,
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "advantages": advantages,
                "vllm_logprobs": vllm_logprobs,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "batch_logprobs": batch_logprobs,
                "micro_idx": micro_idx,
            })

        return prepared
    
    def _loss_func(
        self,
        logits: torch.Tensor,
        batch: dict,
        num_micro_batches: int,
        tracker: GPUTimelineLogger | _NullTracker | None = None,
        backward_cuda_state: dict | None = None,
        minibatch: int = -1,
    ):
        """
        Loss function called by Megatron's pipeline schedule.

        Computes the RL loss (same as FSDP backend) from logits.
        Returns (loss, normalizer, logging_dict) as Megatron expects.

        Args:
            backward_cuda_state: When provided, a mutable dict used to record
                a CUDA start-event right before this function returns.  The
                Megatron scheduler calls ``torch.autograd.backward`` immediately
                after, so the event marks the beginning of the backward pass
                for this microbatch.
            minibatch: Mini batch index (-1 if not applicable or only 1 minibatch).
        """
        from telescope.trainer.loss import compute_rl_loss

        # Get original data from batch (already CP-chunked by _prepare_megatron_data).
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"]
        advantages = batch["advantages"]
        vllm_logprobs = batch.get("vllm_logprobs")
        micro_idx = int(batch.get("micro_idx", -1))

        # Determine ref_logprobs for ratio computation.
        ref_logprobs = None
        if config.cfg.ppo_clip_ref_logprobs == "batch":
            ref_logprobs = batch.get("batch_logprobs")

        def _track(name: str):
            if tracker is not None:
                return tracker.track(name, microbatch=micro_idx, minibatch=minibatch)
            return nullcontext()

        # Megatron logits may be [1, padded_len, vocab] - slice to original length
        orig_len = input_ids.shape[-1]
        if logits.shape[1] > orig_len:
            logits = logits[:, :orig_len, :]

        from megatron.core import parallel_state as mpu

        # Guard algorithms that need full-sequence information — incompatible
        # with CP because each rank only sees its local chunk.
        cp_ws = mpu.get_context_parallel_world_size()
        if cp_ws > 1 and config.cfg.algorithm == "gspo":
            raise NotImplementedError(
                "GSPO requires full-sequence per-sample statistics and is not yet "
                "supported with context parallelism (CP > 1). Use grpo, rloo, "
                "reinforce_pp, dr_grpo, cispo, or sapo instead."
            )

        # With CP, non-zero ranks have position_ids that don't start at 0
        # (e.g. [k, k+1, ...]).  Re-zero them so sample-boundary detection
        # (DR-GRPO token_sum_norm) still works.  The model forward pass has
        # already completed, so this doesn't affect RoPE.
        loss_position_ids = batch.get("position_ids")
        if cp_ws > 1 and loss_position_ids is not None:
            loss_position_ids = loss_position_ids - loss_position_ids[:, :1]

        # Megatron schedule already applies 1/num_microbatches scaling when
        # the loss function returns a 3-tuple, so use num_micro_batches=1
        # here to avoid double scaling the gradients.
        if mpu.get_tensor_model_parallel_world_size() > 1:
            scaled_loss, metrics = _compute_rl_loss_vocab_parallel(
                logits=logits,
                input_ids=input_ids,
                loss_mask=loss_mask,
                advantages=advantages,
                vllm_logprobs=vllm_logprobs,
                tp_group=mpu.get_tensor_model_parallel_group(),
                num_micro_batches=1,
                track=_track,
                position_ids=loss_position_ids,
                ref_logprobs=ref_logprobs,
            )
        else:
            scaled_loss, metrics = compute_rl_loss(
                logits=logits,
                input_ids=input_ids,
                loss_mask=loss_mask,
                advantages=advantages,
                vllm_logprobs=vllm_logprobs,
                num_micro_batches=1,
                track=_track,
                position_ids=loss_position_ids,
                ref_logprobs=ref_logprobs,
            )
        
        # Megatron expects (loss, num_tokens, logging_dict)
        # where logging_dict = {"keys": [...], "values": tensor}
        keys = list(metrics.keys())
        values = torch.tensor(
            [metrics.get("num_tokens", 1)] + [metrics[k] for k in keys],
            device=logits.device,
        )
        
        # num_tokens must be an int Tensor — Megatron does torch.clamp(num_tokens, min=1)
        # and also total_num_tokens += num_tokens where total_num_tokens is int-typed
        num_tokens = torch.tensor(1, dtype=torch.int64, device=logits.device)

        # Record backward start event.  Megatron's backward_step (which calls
        # torch.autograd.backward) runs immediately after this function returns,
        # so this CUDA event marks the start of the backward pass.  The
        # corresponding end-event is recorded either at the start of the next
        # forward_step_func or inside the finalize_model_grads wrapper.
        if backward_cuda_state is not None:
            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            backward_cuda_state["start"] = start_ev
            backward_cuda_state["micro_idx"] = micro_idx

        return scaled_loss, num_tokens, {"keys": keys, "values": values}
    
    def _collect_metrics(
        self,
        losses_reduced: list[dict],
        grad_norm: float,
        num_micro_batches: int,
    ) -> dict:
        """Collect and average metrics from Megatron's training step."""
        from megatron.core import mpu

        metrics = {
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
            "entropy": 0.0,
            "kl_divergence_inference": 0.0,
        }

        if mpu.is_pipeline_last_stage(ignore_virtual=True) and losses_reduced:
            keys = losses_reduced[0].get("keys", [])
            all_values = None
            for entry in losses_reduced:
                vals = entry.get("values")
                if vals is not None:
                    if all_values is None:
                        all_values = vals.clone()
                    else:
                        all_values += vals

            if all_values is not None and keys:
                num_samples = all_values[0].item()
                for i, key in enumerate(keys):
                    if num_samples > 0:
                        metrics[key] = all_values[i + 1].item() / num_micro_batches

        metrics.pop("loss", None)
        metrics.pop("num_tokens", None)
        return metrics

    def gather_weights_for_inference(self) -> dict[str, torch.Tensor]:
        """
        Gather full model weights in HuggingFace format for inference sync.

        Tries bridge-based export first (model-agnostic), then falls back to
        manual all-gather + conversion (Qwen-specific).

        ALL ranks must call this (collective operations inside).
        Only dp_rank=0, tp_rank=0 produces meaningful results.

        Memory optimizations (configurable):
        - ``weight_broadcast_cpu_staging``: stage gathered weights to CPU to
          avoid accumulating the full HF state dict on GPU (~28 GB for 14B).
        - ``weight_broadcast_pin_memory``: use pinned CPU memory for ~2-3x
          faster CPU↔GPU transfers during broadcast.
        - ``weight_broadcast_free_grad_buffers``: temporarily release Megatron
          DDP grad buffer storage (~14 GB) during the gather/broadcast window.
        """
        cpu_stage = bool(config.cfg.weight_broadcast_cpu_staging)
        pin_memory = bool(config.cfg.weight_broadcast_pin_memory)
        free_grads = bool(config.cfg.weight_broadcast_free_grad_buffers)

        # Temporarily free grad buffer storage to reclaim GPU memory.
        _grad_buf_sizes = _free_grad_buffers(self._model) if free_grads else None
        if _grad_buf_sizes is not None:
            torch.cuda.empty_cache()

        try:
            if self._bridge is not None:
                try:
                    result = self._gather_weights_via_bridge(cpu_stage, pin_memory)
                    return result
                except Exception as e:
                    _log.warning(
                        f"Bridge weight export failed ({e}), falling back to manual. "
                        f"Bridge disabled for future calls.",
                        rank=self._rank,
                    )
                    self._bridge = None
            return self._gather_weights_manual(cpu_stage, pin_memory)
        finally:
            if _grad_buf_sizes is not None:
                _restore_grad_buffers(self._model, _grad_buf_sizes)

    def _gather_weights_via_bridge(
        self, cpu_stage: bool, pin_memory: bool,
    ) -> dict[str, torch.Tensor]:
        """Export weights via megatron.bridge (model-agnostic).

        The bridge handles TP all-gather, name mapping, QKV/GLU unfusion, and
        vocab padding removal internally.  All TP ranks must call this since
        the bridge uses collective operations.

        PP-aware: with PP>1, each stage exports its own layers via the bridge,
        then ``_gather_pp_state_dicts`` merges them onto pp_rank=0.

        EP-aware: with EP>1, expert parameters from each EP rank are gathered
        onto ep_rank=0 after bridge export.
        """
        from megatron.core import parallel_state as mpu

        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()

        # EP info
        try:
            ep_size = mpu.get_expert_model_parallel_world_size()
            ep_rank = mpu.get_expert_model_parallel_rank()
            ep_group = mpu.get_expert_model_parallel_group()
        except Exception:
            ep_size = 1
            ep_rank = 0
            ep_group = None

        hf_config = self._hf_config
        num_moe_experts = getattr(hf_config, "num_experts", None) or \
                          getattr(hf_config, "num_local_experts", None)
        num_local_experts = (num_moe_experts // ep_size) if (num_moe_experts and ep_size > 1) else None

        # Bridge reads share_embeddings_and_output_weights from the model config.
        # Megatron Core stores it on the GPTModel, not on TransformerConfig, so
        # we temporarily patch it (same pattern as MILES).
        try:
            from megatron.core.utils import unwrap_model
            unwrapped = unwrap_model(self._model)[0]
        except (ImportError, Exception):
            unwrapped = self._model[0]
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

        model_config = unwrapped.config
        patched = not hasattr(model_config, "share_embeddings_and_output_weights")
        if patched:
            model_config.share_embeddings_and_output_weights = getattr(
                unwrapped, "share_embeddings_and_output_weights", False,
            )

        try:
            # export_hf_weights is collective — all TP ranks participate.
            # Returns iterator of (hf_name, weight, ...) tuples.
            named_weights = self._bridge.export_hf_weights(
                self._model, cpu=cpu_stage,
            )

            hf_state_dict: dict[str, torch.Tensor] = {}
            if tp_rank == 0:
                for item in named_weights:
                    hf_name, weight = item[0], item[1]
                    if cpu_stage and isinstance(weight, torch.Tensor) and weight.is_cuda:
                        if pin_memory:
                            pinned = torch.empty(
                                weight.shape, dtype=weight.dtype, pin_memory=True,
                            )
                            pinned.copy_(weight)
                            hf_state_dict[hf_name] = pinned
                        else:
                            hf_state_dict[hf_name] = weight.detach().cpu()
                        del weight
                    else:
                        hf_state_dict[hf_name] = weight
            else:
                # Non-tp_rank=0: consume generator to participate in collectives
                for _ in named_weights:
                    pass
        finally:
            if patched:
                delattr(model_config, "share_embeddings_and_output_weights")

        # With PP>1, remap local layer indices to global before merging.
        # The bridge exports HF names with PP-local layer indices (e.g.,
        # pp_rank=1 exports "model.layers.0" instead of "model.layers.18").
        # Without remapping, _gather_pp_state_dicts would see duplicate keys
        # from different stages and the last stage would silently overwrite
        # the first stage's layers.
        if pp_size > 1 and tp_rank == 0 and pp_rank > 0:
            num_layers = self._hf_config.num_hidden_layers
            layers_per_stage = num_layers // pp_size
            layer_offset = pp_rank * layers_per_stage

            # Detect whether the bridge already used global indices by
            # checking if any exported key references a layer >= layer_offset.
            _already_global = any(
                int(m.group(1)) >= layer_offset
                for k in hf_state_dict
                for m in [re.search(r"model\.layers\.(\d+)", k)]
                if m is not None
            )
            if not _already_global:
                remapped: dict[str, torch.Tensor] = {}
                for hf_name, weight in hf_state_dict.items():
                    remapped[_remap_hf_layer_index(hf_name, layer_offset)] = weight
                hf_state_dict = remapped

        # With EP>1, remap local expert indices to global and gather expert
        # params from all EP ranks onto ep_rank=0.
        if ep_size > 1 and tp_rank == 0 and num_local_experts and ep_group is not None:
            if ep_rank > 0:
                remapped: dict[str, torch.Tensor] = {}
                for hf_name, weight in hf_state_dict.items():
                    remapped[_remap_hf_expert_index(hf_name, ep_rank, num_local_experts)] = weight
                hf_state_dict = remapped
            hf_state_dict = _gather_ep_expert_params(
                hf_state_dict, ep_rank, ep_size, ep_group, cpu_stage,
            )

        # With PP>1, gather state dicts from all PP stages onto pp_rank=0.
        if pp_size > 1 and tp_rank == 0:
            hf_state_dict = _gather_pp_state_dicts(
                hf_state_dict, pp_rank, pp_size, mpu, cpu_stage,
            )

        if cpu_stage and tp_rank == 0:
            torch.cuda.empty_cache()

        return hf_state_dict

    def _gather_weights_manual(
        self, cpu_stage: bool, pin_memory: bool,
    ) -> dict[str, torch.Tensor]:
        """Export weights via manual all-gather + Qwen-specific conversion.

        Fallback path when megatron.bridge is not available.

        PP-aware: with PP>1, each stage gathers its own TP-sharded layers
        into HF format (with global layer indices), then sends them to
        pp_rank=0 which merges the full state dict.

        EP-aware: with EP>1, each EP rank holds a different subset of experts.
        Local expert indices are remapped to global, then expert parameters
        are gathered from all EP ranks onto ep_rank=0.
        """
        from megatron.core import parallel_state as mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        hf_config = self._hf_config

        # EP info
        try:
            ep_size = mpu.get_expert_model_parallel_world_size()
            ep_rank = mpu.get_expert_model_parallel_rank()
            ep_group = mpu.get_expert_model_parallel_group()
        except Exception:
            ep_size = 1
            ep_rank = 0
            ep_group = None

        num_moe_experts = getattr(hf_config, "num_experts", None) or \
                          getattr(hf_config, "num_local_experts", None)
        num_local_experts = (num_moe_experts // ep_size) if (num_moe_experts and ep_size > 1) else None

        # Compute global layer offset for this PP stage.
        num_layers = hf_config.num_hidden_layers
        layers_per_stage = num_layers // pp_size
        layer_offset = pp_rank * layers_per_stage

        hf_state_dict: dict[str, torch.Tensor] = {}

        for model_chunk in self._model:
            for name, param in model_chunk.named_parameters():
                if not name.startswith("module.module."):
                    name = "module." + name

                full_param = _all_gather_tp_param(name, param, tp_group)

                if tp_rank == 0:
                    if "output_layer.weight" in name or "word_embeddings.weight" in name:
                        full_param = full_param[:hf_config.vocab_size, :]

                    # Remap local layer indices to global indices for PP>1.
                    export_name = name
                    if pp_size > 1 and layer_offset > 0:
                        export_name = _remap_layer_index(name, layer_offset)

                    # Remap local expert indices to global indices for EP>1.
                    if ep_size > 1 and num_local_experts and "local_experts" in export_name:
                        export_name = _remap_expert_index(export_name, ep_rank, num_local_experts)

                    try:
                        hf_params = _convert_qwen_to_hf(
                            name=export_name,
                            param=full_param,
                            num_attention_heads=hf_config.num_attention_heads,
                            num_key_value_heads=getattr(
                                hf_config, "num_key_value_heads",
                                hf_config.num_attention_heads,
                            ),
                            hidden_size=hf_config.hidden_size,
                            head_dim=getattr(hf_config, "head_dim", None),
                        )
                        for hf_name, hf_param in hf_params:
                            if cpu_stage and isinstance(hf_param, torch.Tensor) and hf_param.is_cuda:
                                if pin_memory:
                                    pinned = torch.empty(
                                        hf_param.shape, dtype=hf_param.dtype, pin_memory=True,
                                    )
                                    pinned.copy_(hf_param)
                                    hf_state_dict[hf_name] = pinned
                                else:
                                    hf_state_dict[hf_name] = hf_param.detach().cpu()
                                del hf_param
                            else:
                                hf_state_dict[hf_name] = hf_param
                    except ValueError:
                        _log.warning(f"Skipping unknown param for HF conversion: {name}")

                del full_param

        # With EP>1, gather expert params from all EP ranks onto ep_rank=0.
        if ep_size > 1 and tp_rank == 0 and ep_group is not None:
            hf_state_dict = _gather_ep_expert_params(
                hf_state_dict, ep_rank, ep_size, ep_group, cpu_stage,
            )

        # With PP>1, gather state dicts from all PP stages onto pp_rank=0.
        if pp_size > 1 and tp_rank == 0:
            hf_state_dict = _gather_pp_state_dicts(
                hf_state_dict, pp_rank, pp_size, mpu, cpu_stage,
            )

        if cpu_stage and tp_rank == 0:
            torch.cuda.empty_cache()

        return hf_state_dict

    def barrier(self) -> None:
        """Synchronize all ranks."""
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
        """Data parallel rank (accounts for TP/PP/CP grouping)."""
        try:
            from megatron.core import mpu
            return mpu.get_data_parallel_rank(with_context_parallel=False)
        except Exception:
            return self._rank

    @property
    def dp_world_size(self) -> int:
        """Number of data parallel replicas."""
        try:
            from megatron.core import mpu
            return mpu.get_data_parallel_world_size(with_context_parallel=False)
        except Exception:
            tp = config.cfg.megatron_tensor_parallel_size
            pp = config.cfg.megatron_pipeline_parallel_size
            cp = config.cfg.megatron_context_parallel_size
            return self._world_size // (tp * pp * cp)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_weight_broadcast_rank(self) -> bool:
        """
        Only (dp_rank=0, tp_rank=0, pp_rank=0, ep_rank=0) broadcasts weights
        to inference. With PP>1, gather_weights_for_inference merges all PP
        stages' weights onto pp_rank=0 before broadcast. With EP>1, expert
        weights are gathered onto ep_rank=0 before broadcast.
        """
        try:
            from megatron.core import mpu
            is_dp0 = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            is_tp0 = mpu.get_tensor_model_parallel_rank() == 0
            is_pp0 = mpu.get_pipeline_model_parallel_rank() == 0
            try:
                is_ep0 = mpu.get_expert_model_parallel_rank() == 0
            except Exception:
                is_ep0 = True
            return is_dp0 and is_tp0 and is_pp0 and is_ep0
        except Exception:
            return self._rank == 0

    @property
    def model(self):
        """Access the underlying model chunks."""
        return self._model

    @property
    def optimizer(self):
        """Access the optimizer."""
        return self._optimizer

    @property
    def scheduler(self):
        """Access the LR scheduler (None if lr_scheduler='none')."""
        return self._scheduler

    def _get_rng_state(self):
        """Collect RNG state for this rank, wrapped as a ShardedObject for DCP.

        Each (PP, TP) coordinate saves one list of RNG-state dicts (one entry
        per DP rank when ``data_parallel_random_init`` is true, otherwise a
        single-element list).  DP replicas share the same (PP, TP) slot via
        ``replica_id``.
        """
        from megatron.core import parallel_state as mpu, tensor_parallel
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        rng_state_list = [rng_state]

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)

        return ShardedObject(
            "rng_state",
            rng_state_list,
            (pp_size, tp_size),
            (pp_rank, tp_rank),
            replica_id=dp_rank,
        )

    def _load_rng_state(self, rng_state_list):
        """Restore RNG state from a loaded checkpoint entry."""
        from megatron.core import tensor_parallel

        # rng_state_list is a list with one dict (no data_parallel_random_init)
        rng_state = rng_state_list[0]

        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])

        if rng_state.get("rng_tracker_states"):
            tensor_parallel.get_cuda_rng_tracker().set_states(
                rng_state["rng_tracker_states"]
            )

    def save_checkpoint(self, step: int, ckpt_dir: "Path", tracker=None) -> None:
        """Save Megatron checkpoint using megatron.core.dist_checkpointing. All ranks must call together."""
        from pathlib import Path
        from megatron.core import dist_checkpointing

        def _track(name: str):
            if tracker is not None:
                return tracker.track(name, cpu=True)
            return nullcontext()

        ckpt_dir = Path(ckpt_dir)
        tmp_path = ckpt_dir / f"step_{step}_tmp"
        final_path = ckpt_dir / f"step_{step}"

        with _track("mkdir_barrier"):
            if self._rank == 0:
                tmp_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()

        start = time.perf_counter()
        if self._rank == 0:
            _log.info(f"Saving Megatron checkpoint at step {step}", step=step, rank=self._rank)

        with _track("get_state_dict"):
            # Build sharded state dict from model chunks
            sharded_state_dict = {}
            for vpp_rank, model_chunk in enumerate(self._model):
                key = f"model{vpp_rank}" if len(self._model) > 1 else "model"
                inner = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
                sharded_state_dict[key] = inner.sharded_state_dict()

            # Add optimizer state (only when saving full training state for resume)
            save_training_state = config.cfg.checkpoint_save_training_state
            if save_training_state:
                sharded_state_dict["optimizer"] = self._optimizer.sharded_state_dict(
                    sharded_state_dict
                )
                if self._scheduler is not None:
                    sharded_state_dict["scheduler"] = self._scheduler.state_dict()
                sharded_state_dict["rng_state"] = self._get_rng_state()
            sharded_state_dict["step"] = step

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with _track("dist_checkpointing_save"):
                    dist_checkpointing.save(sharded_state_dict, str(tmp_path))
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
                        import shutil as _shutil
                        if tmp_path.exists():
                            _shutil.rmtree(tmp_path)
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

                from megatron.core import parallel_state as _mpu
                meta = {
                    "base_model": config.cfg.model,
                    "step": step,
                    "backend": "megatron",
                    "vocab_size": self._hf_config.vocab_size,
                    "padded_vocab_size": self._padded_vocab_size,
                    "tp_size": _mpu.get_tensor_model_parallel_world_size(),
                    "pp_size": _mpu.get_pipeline_model_parallel_world_size(),
                    "ep_size": getattr(_mpu, "get_expert_model_parallel_world_size", lambda: 1)(),
                }
                meta_path = tmp_path / "meta.json"
                meta_path.write_text(json.dumps(meta, indent=2))

        # Pseudo-atomic: rank 0 renames tmp -> final after all ranks finish writing
        with _track("barrier_rename"):
            dist.barrier()
            if self._rank == 0:
                if final_path.exists():
                    import shutil
                    shutil.rmtree(final_path)
                tmp_path.rename(final_path)
            dist.barrier()

        elapsed = time.perf_counter() - start
        if self._rank == 0:
            _log.timing("Megatron checkpoint saved", elapsed, step=step, rank=self._rank)

    def load_checkpoint(self, step: int, ckpt_dir: "Path") -> None:
        """Load Megatron checkpoint using megatron.core.dist_checkpointing. All ranks must call together."""
        from pathlib import Path
        from megatron.core import dist_checkpointing

        ckpt_dir = Path(ckpt_dir)
        ckpt_path = ckpt_dir / f"step_{step}"

        start = time.perf_counter()
        if self._rank == 0:
            _log.info(f"Loading Megatron checkpoint from step {step}", step=step, rank=self._rank)

        # Build sharded state dict structure matching what was saved
        sharded_state_dict = {}
        for vpp_rank, model_chunk in enumerate(self._model):
            key = f"model{vpp_rank}" if len(self._model) > 1 else "model"
            inner = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
            sharded_state_dict[key] = inner.sharded_state_dict()

        # Initialize optimizer states with dummy values so that
        # sharded_state_dict() can read the step counter (state_dict()
        # asserts exactly one step exists).  The dummy values are
        # overwritten in-place by dist_checkpointing.load() below.
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

        for opt in getattr(self._optimizer, "chained_optimizers", [self._optimizer]):
            if isinstance(opt, DistributedOptimizer):
                opt._init_optimizer_states_with_dummy_values()

        sharded_state_dict["optimizer"] = self._optimizer.sharded_state_dict(
            sharded_state_dict, is_loading=True
        )
        sharded_state_dict["rng_state"] = self._get_rng_state()
        sharded_state_dict["step"] = 0

        # Load fills tensors in-place through sharded state dict references
        dist_checkpointing.load(sharded_state_dict, str(ckpt_path))

        # Restore scheduler state if present
        if self._scheduler is not None and "scheduler" in sharded_state_dict:
            self._scheduler.load_state_dict(sharded_state_dict["scheduler"])

        # Restore RNG state for deterministic resume
        if "rng_state" in sharded_state_dict:
            rng_data = sharded_state_dict["rng_state"]
            # After DCP load, ShardedObject is replaced with its deserialized data
            if isinstance(rng_data, list):
                self._load_rng_state(rng_data)
            elif hasattr(rng_data, "data"):
                self._load_rng_state(rng_data.data)
            else:
                if self._rank == 0:
                    _log.warning("RNG state found but format not recognized, skipping restore")

        dist.barrier()
        elapsed = time.perf_counter() - start
        if self._rank == 0:
            _log.timing("Megatron checkpoint loaded", elapsed, step=step, rank=self._rank)

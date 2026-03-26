"""
Shared RL loss computation.

This module contains the core RL loss function used by all training backends.
The loss supports multiple algorithms (grpo, cispo, gspo, sapo, rloo,
reinforce_pp, dr_grpo) selected via config.cfg.algorithm, plus algorithm-agnostic
modifiers: PPO clip (trust-region clipping), dual-clip, TIS (Truncated Importance
Sampling / ICE-POP), KL penalty (k2/k3), entropy bonus, and sequence-level IS masking.

Both FSDP and Megatron backends call these functions with their computed
log-probabilities to get the loss and metrics.
"""
from collections.abc import Callable
from contextlib import nullcontext

import torch

from telescope.utils import config


def _masked_mean_per_sample(
    x: torch.Tensor,
    mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute masked mean per packed sample using position_ids boundaries.

    Returns tensor same shape as x, with each sample's tokens replaced
    by that sample's masked mean.
    """
    # Detect sample starts: position_ids == 0
    starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0]  # [num_samples]
    result = torch.zeros_like(x)

    for i in range(len(starts)):
        s = starts[i].item()
        e = starts[i + 1].item() if i + 1 < len(starts) else x.shape[-1]
        seg_x = x[0, s:e]
        seg_m = mask[0, s:e]
        denom = seg_m.sum().clamp_min(1)
        mean_val = (seg_x * seg_m).sum() / denom
        result[0, s:e] = mean_val

    return result


def _masked_count_per_sample(
    mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Count valid (masked) tokens per packed sample, broadcast to each position.

    Returns tensor same shape as mask, with each sample's tokens set to that
    sample's valid-token count.
    """
    starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0]
    result = torch.ones_like(mask, dtype=torch.float)

    for i in range(len(starts)):
        s = starts[i].item()
        e = starts[i + 1].item() if i + 1 < len(starts) else mask.shape[-1]
        count = mask[0, s:e].sum().clamp_min(1).float()
        result[0, s:e] = count

    return result


def _compute_kl_penalty(
    vllm_delta: torch.Tensor,
    estimator: str,
) -> torch.Tensor:
    """Compute per-token KL penalty using the configured estimator.

    Args:
        vllm_delta: log(pi_curr / pi_rollout) per token.
        estimator: "k2" or "k3".

    Returns:
        Non-negative KL penalty per token (same shape as vllm_delta).
    """
    if estimator == "k2":
        return vllm_delta.pow(2)
    # k3: exp(-delta) - 1 + delta  (Schulman's lower-variance, non-negative estimator)
    # Input clamp prevents exp() overflow; output clamp caps extreme divergence values.
    clamped = vllm_delta.clamp(-20.0, 20.0)
    return (torch.exp(-clamped) - 1.0 + clamped).clamp(max=10.0)


def _apply_seq_is_masking(
    shift_loss_mask: torch.Tensor,
    vllm_delta: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Zero loss mask for sequences whose geometric-mean IS ratio is outside bounds.

    Computes exp(mean_per_sample(log(pi_curr/pi_rollout))) and masks sequences
    where this value falls outside [seq_is_mask_low, seq_is_mask_high].

    Returns a (possibly modified) copy of shift_loss_mask.
    """
    mask_2d = shift_loss_mask.unsqueeze(0) if shift_loss_mask.dim() == 1 else shift_loss_mask

    if position_ids is not None:
        # Packed batch: use position_ids to find sequence boundaries
        log_mean = _masked_mean_per_sample(
            vllm_delta.unsqueeze(0) if vllm_delta.dim() == 1 else vllm_delta,
            mask_2d,
            position_ids,
        )
        geo_mean = torch.exp(log_mean.squeeze(0).clamp(-20.0, 20.0))
    else:
        # Single sequence: one geometric mean for the whole batch dim
        denom = mask_2d.sum(dim=-1, keepdim=True).clamp_min(1)
        log_mean = (vllm_delta * shift_loss_mask).sum(dim=-1, keepdim=True) / denom
        geo_mean = torch.exp(log_mean.clamp(-20.0, 20.0)).expand_as(shift_loss_mask)

    in_bounds = (geo_mean >= config.cfg.seq_is_mask_low) & (geo_mean <= config.cfg.seq_is_mask_high)
    return shift_loss_mask * in_bounds.to(shift_loss_mask.dtype)


def compute_rl_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
    vllm_logprobs: torch.Tensor | None,
    num_micro_batches: int,
    track: Callable[[str], object] | None = None,
    position_ids: torch.Tensor | None = None,
    ref_logprobs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute RL policy gradient loss and metrics for a single micro batch.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        input_ids: Input token IDs [batch, seq_len]
        loss_mask: Binary mask for response tokens [batch, seq_len]
        advantages: Per-token advantages [batch, seq_len] or [batch, seq_len, 1]
        vllm_logprobs: Log probs from inference (for PPO/TIS) [batch, seq_len] or None
        num_micro_batches: Total number of micro batches (for gradient accumulation scaling)
        track: Optional callback returning a context manager for timeline event tracking.
        position_ids: Position IDs [batch, seq_len] for packed sequence boundary detection (GSPO).
        ref_logprobs: Reference logprobs for ratio computation (use_ppo_clip, cispo, gspo, sapo).
            When None, falls back to vllm_logprobs. vllm_logprobs is still used for TIS and metrics.

    Returns:
        Tuple of:
            - scaled_loss: Loss tensor (scaled for gradient accumulation), suitable for .backward()
            - metrics: Dict with loss, entropy, kl_divergence_inference, num_tokens
    """
    def _track(event_name: str):
        if track is not None:
            return track(event_name)
        return nullcontext()

    with _track("prepare_tensors"):
        # Reshape advantages if needed
        if advantages.dim() == 2:
            advantages = advantages.unsqueeze(-1)

        # Shift logits/labels for next-token prediction
        shift_logits = logits[..., :-1, :] / config.cfg.get_sampling_params()["temperature"]
        labels = input_ids[:, 1:]
        shift_loss_mask = loss_mask[..., 1:]
        shift_advantages = advantages[..., 1:, :]

    entropy_value = None  # set in loss block if entropy_coef > 0, else in metrics block

    with _track("loss_computation"):
        # Compute log probs of chosen tokens
        log_probs = shift_logits.log_softmax(dim=-1)
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # Delta for ratio computation (use_ppo_clip, cispo, gspo, sapo).
        # Uses ref_logprobs when provided, otherwise falls back to vllm_logprobs.
        _ref = ref_logprobs if ref_logprobs is not None else vllm_logprobs
        ratio_delta = None
        if _ref is not None:
            ratio_delta = log_probs - _ref[..., 1:]

        # Delta against rollout (vLLM) logprobs — always used for TIS and metrics
        # (measures distance from the rollout policy).
        vllm_delta = None
        if vllm_logprobs is not None:
            vllm_delta = log_probs - vllm_logprobs[..., 1:]

        # --- Sequence-level IS masking (off-policy rejection) ---
        _pre_mask_tokens = shift_loss_mask.sum().item()  # for metric
        if config.cfg.seq_is_masking and vllm_delta is not None:
            shift_loss_mask = _apply_seq_is_masking(shift_loss_mask, vllm_delta, position_ids)

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

        # --- Dual-clip PPO (cap loss magnitude for negative advantages) ---
        _dual_clip_frac = 0.0  # for metric
        if config.cfg.clip_ratio_c is not None:
            c = config.cfg.clip_ratio_c
            pg_loss3 = torch.sign(adv) * c * adv  # = c * |adv|, always positive
            _dual_clip_active = (pg_loss3.detach() < policy_loss.detach()) & (shift_loss_mask > 0)
            _dual_clip_frac = float(_dual_clip_active.sum().item() / max(shift_loss_mask.sum().item(), 1))
            policy_loss = torch.min(policy_loss, pg_loss3)

        # --- TIS correction (always uses vllm_delta = distance from rollout policy) ---
        _tis_mean = 0.0  # for metric
        _tis_zero_frac = 0.0  # for metric (ICE-POP)
        if config.cfg.use_tis and vllm_delta is not None:
            logprob_diff = vllm_delta.clamp(
                -config.cfg.tis_logprob_clamp, config.cfg.tis_logprob_clamp
            )
            if config.cfg.tis_mode == "icepop":
                # ICE-POP: zero tokens outside [floor, cap] instead of clamping
                tis_ratio = torch.exp(logprob_diff).detach()
                in_bounds = (tis_ratio >= config.cfg.tis_floor) & (tis_ratio <= config.cfg.tis_cap)
                _valid = shift_loss_mask.sum().clamp_min(1)
                _tis_zero_frac = float(((~in_bounds) & (shift_loss_mask > 0)).sum().item() / _valid.item())
                tis_ratio = tis_ratio * in_bounds.to(tis_ratio.dtype)
            else:
                # Standard truncate mode: clamp IS weights
                tis_ratio = torch.exp(logprob_diff).clamp(max=config.cfg.tis_cap).detach()
            _tis_mean = float((tis_ratio * shift_loss_mask).sum().item() / shift_loss_mask.sum().clamp_min(1).item())
            policy_loss = policy_loss * tis_ratio

        # --- KL penalty ---
        if config.cfg.kl_penalty_tau > 0 and vllm_delta is not None:
            kl_penalty = config.cfg.kl_penalty_tau * _compute_kl_penalty(
                vllm_delta, config.cfg.kl_estimator
            )
            policy_loss = policy_loss + kl_penalty

        # --- Apply mask and compute loss ---
        policy_loss = policy_loss * shift_loss_mask
        num_tokens = int(shift_loss_mask.sum().item())

        if algo == "dr_grpo" and config.cfg.dr_grpo_loss_agg_mode == "token_sum_norm":
            # DR-GRPO paper §3.2: normalize by (num_samples × seq_len) instead of
            # total valid tokens, removing response-level length bias.
            if position_ids is not None:
                starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0]
                num_samples = len(starts)
                # Padding appended by pad_micro_batch starts with position_id=0,
                # creating a ghost segment with no trainable tokens.  Exclude it.
                if num_samples > 1 and not loss_mask[0, starts[-1]:].any():
                    num_samples -= 1
            else:
                num_samples = 1
            denom = float(num_samples * shift_loss_mask.shape[-1])
            loss = policy_loss.sum() / max(denom, 1.0)
        else:
            denom = shift_loss_mask.sum().clamp_min(1).float()
            loss = policy_loss.sum() / denom

        # --- Entropy bonus (subtract entropy from loss to prevent policy collapse) ---
        if config.cfg.entropy_coef > 0:
            _full_log_p = shift_logits.float().log_softmax(dim=-1)
            _ent = -(_full_log_p.exp() * _full_log_p).sum(dim=-1)  # [B, S]
            _mask_denom = shift_loss_mask.sum().clamp_min(1).float()
            _ent_mean = (_ent * shift_loss_mask).sum() / _mask_denom
            loss = loss - config.cfg.entropy_coef * _ent_mean
            entropy_value = float(_ent_mean.detach().item())
            del _full_log_p

        # Scale for gradient accumulation
        scaled_loss = loss / num_micro_batches
        loss_value = float(loss.item())

    # --- Metrics ---
    with _track("compute_entropy"):
        if entropy_value is None:
            entropy_value = _compute_entropy(shift_logits, shift_loss_mask)

    with _track("compute_kl"):
        kl_value = 0.0
        if vllm_delta is not None:
            kl_value = _compute_kl_from_logprob_diff(vllm_delta, shift_loss_mask)

    metrics = {
        "loss": loss_value,
        "entropy": entropy_value,
        "kl_divergence_inference": kl_value,
        "num_tokens": num_tokens,
    }
    if config.cfg.kl_penalty_tau > 0 and vllm_delta is not None:
        kl_pen = _compute_kl_penalty(vllm_delta.detach(), config.cfg.kl_estimator)
        kl_pen = (kl_pen * shift_loss_mask).sum() / shift_loss_mask.sum().clamp_min(1)
        metrics["kl_penalty"] = float(kl_pen.item())
    if config.cfg.seq_is_masking:
        _post_mask_tokens = shift_loss_mask.sum().item()
        metrics["seq_is_masked_frac"] = 1.0 - _post_mask_tokens / max(_pre_mask_tokens, 1)
    if config.cfg.clip_ratio_c is not None:
        metrics["dual_clip_frac"] = _dual_clip_frac
    if config.cfg.use_tis:
        metrics["tis_mean"] = _tis_mean
        if config.cfg.tis_mode == "icepop":
            metrics["tis_zero_frac"] = _tis_zero_frac
    if config.cfg.entropy_coef > 0:
        metrics["entropy_bonus"] = config.cfg.entropy_coef * entropy_value

    return scaled_loss, metrics


def _compute_entropy(shift_logits: torch.Tensor, shift_loss_mask: torch.Tensor) -> float:
    """Compute mean entropy of the policy over masked positions.

    When ``config.cfg.entropy_chunk_size > 0``, positions are processed in chunks
    to avoid materializing full [batch*seq, vocab] softmax tensors which
    easily OOM on large vocabs.  Set to 0 to disable chunking.
    """
    chunk_size = config.cfg.entropy_chunk_size
    with torch.no_grad():
        if chunk_size <= 0:
            # Unchunked: fast but allocates full [B*S, V] tensors.
            log_p = torch.log_softmax(shift_logits.float(), dim=-1)
            entropy = -(log_p.exp() * log_p).sum(dim=-1)  # [B, S]
            masked = entropy * shift_loss_mask
            return float((masked.sum() / shift_loss_mask.sum().clamp_min(1)).item())

        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(-1, V)   # [B*S, V]
        flat_mask = shift_loss_mask.reshape(-1)      # [B*S]

        weighted_sum = torch.tensor(0.0, device=shift_logits.device)
        mask_sum = flat_mask.sum()

        for start in range(0, flat_logits.size(0), chunk_size):
            chunk_logits = flat_logits[start : start + chunk_size].float()
            chunk_mask = flat_mask[start : start + chunk_size]

            log_p = torch.log_softmax(chunk_logits, dim=-1)
            entropy = -(log_p.exp() * log_p).sum(dim=-1)  # [chunk]
            weighted_sum += (entropy * chunk_mask).sum()

        return float((weighted_sum / mask_sum.clamp_min(1)).item())


def _compute_kl_from_logprob_diff(
    logprob_diff: torch.Tensor,
    shift_loss_mask: torch.Tensor,
    kl_type: str = "k3",
) -> float:
    """Compute approximate KL from precomputed (log_probs - ref_log_probs)."""
    with torch.no_grad():
        if kl_type == "k1":
            kl = -logprob_diff
        elif kl_type == "k2":
            kl = 0.5 * logprob_diff.pow(2)
        elif kl_type == "k3":
            ratio = torch.exp(logprob_diff)
            kl = (ratio - 1) - logprob_diff
        else:
            raise ValueError(f"Unknown KL type: {kl_type}")

        masked_kl = kl * shift_loss_mask
        total = shift_loss_mask.sum().clamp_min(1)
        return float((masked_kl.sum() / total).item())

"""Batch processing for preparing trainer data with micro batching and sequence packing."""

import math

from telescope.utils import config
from telescope.utils.tlog import get_logger
from telescope.trainer.micro_batch import (
    prepare_sample,
    pack_samples_into_micro_batches,
    pad_micro_batch,
    distribute_micro_batches,
    micro_batches_to_tensors,
)

_log = get_logger("orchestrator")


def preprocess_batch(
    all_groups_results: list[dict],
    num_ranks: int,
    pad_token_id: int = 0,
    start_sample_idx: int = 0,
) -> list[dict]:
    """
    Process rollout results into trainer data with sequence packing.
    
    This function:
    1. Explodes groups into individual samples
    2. Packs samples into micro batches using First Fit Decreasing algorithm
    3. Distributes micro batches across ranks
    4. Each rank may receive multiple micro batches for gradient accumulation
    
    Args:
        all_groups_results: List of group results from rollout
        num_ranks: Number of trainer ranks to split data across
        pad_token_id: Token ID used for padding
        start_sample_idx: Starting sample_idx for run-wide unique IDs
        
    Returns:
        List of trainer data dicts, one per rank
    """
    # Get config values
    seq_len = config.cfg.seq_len
    pad_to_multiple_of = config.cfg.pad_to_multiple_of

    # Context parallelism requires sequences divisible by cp_size for chunking.
    cp_size = config.cfg.megatron_context_parallel_size
    if cp_size > 1:
        pad_to_multiple_of = math.lcm(pad_to_multiple_of, cp_size)

    # Explode groups into per-sample list
    samples = _explode_groups(all_groups_results)
    _normalize_advantages_batch_level(samples)

    # Assign run-wide unique sample_idx to each sample
    for idx, sample in enumerate(samples):
        sample["sample_idx"] = start_sample_idx + idx

    # Prepare samples for packing
    prepared_samples = []
    for sample in samples:
        prepared = prepare_sample(
            prompt_ids=sample["prompt_ids"],
            completion_ids=sample["comp_ids"],
            advantage=sample["advantage"],
            vllm_logprobs=sample.get("vllm_logprobs", []),
            max_seq_len=seq_len,
            completion_mask=sample.get("comp_mask"),
        )
        prepared_samples.append(prepared)

    # Pack samples into micro batches
    micro_batches = pack_samples_into_micro_batches(prepared_samples, seq_len)
    
    # Pad each micro batch
    micro_batches = [
        pad_micro_batch(mb, pad_to_multiple_of, pad_token_id)
        for mb in micro_batches
    ]
    
    # Log packing statistics
    total_samples = len(samples)
    total_micro_batches = len(micro_batches)
    total_tokens = sum(len(mb) for mb in micro_batches)
    total_sample_tokens = sum(len(p) for p in prepared_samples)
    packing_efficiency = total_sample_tokens / total_tokens * 100 if total_tokens > 0 else 0
    _log.debug(
        f"Packing: {total_samples} samples -> {total_micro_batches} micro batches "
        f"| {total_sample_tokens} sample tokens / {total_tokens} total tokens "
        f"| {packing_efficiency:.1f}% efficiency"
    )
    
    # Distribute across ranks
    batches_per_rank = distribute_micro_batches(micro_batches, num_ranks)
    
    # Convert to trainer data format
    trainer_data_per_rank = []
    for rank_idx in range(num_ranks):
        rank_micro_batches = batches_per_rank[rank_idx]
        
        # Convert micro batches to tensors
        tensor_micro_batches = micro_batches_to_tensors(rank_micro_batches)
        
        rank_data = {
            "micro_batches": tensor_micro_batches,
            "num_micro_batches": len(tensor_micro_batches),
        }
        
        trainer_data_per_rank.append(rank_data)

    return trainer_data_per_rank


def _normalize_advantages_batch_level(samples: list[dict]) -> None:
    """Re-normalize advantages across the full batch when ADVANTAGE_NORM='batch' or ALGORITHM='reinforce_pp'."""
    algo = config.cfg.algorithm
    if algo not in ("reinforce_pp",) and config.cfg.advantage_norm != "batch":
        return
    if algo in ("rloo", "dr_grpo"):
        return  # these have their own per-group semantics

    import numpy as np
    all_adv = np.array([s["advantage"] for s in samples])
    mean = all_adv.mean()

    # Batch-level whitening (REINFORCE++ arXiv:2501.03262 / generic batch norm)
    std = (all_adv.std(ddof=1) if len(all_adv) > 1 else 0.0) + 1e-4
    for i, s in enumerate(samples):
        s["advantage"] = float((all_adv[i] - mean) / std)


def _explode_groups(all_groups_results: list[dict]) -> list[dict]:
    """Expand group results into individual samples."""
    samples = []
    
    for group in all_groups_results:
        n = len(group["completion_token_ids"])
        prompt_text = group.get("prompt_text", "")
        env_name = group.get("env_name", "")
        group_id = group.get("group_id", -1)
        completion_texts = group.get("completion_texts", [""] * n)
        vllm_logprobs_list = group.get("vllm_logprobs", [[] for _ in range(n)])
        sample_metrics_list = group.get("sample_metrics", [{} for _ in range(n)])
        golden_answers_list = group.get("golden_answers", [{} for _ in range(n)])
        turns_list = group.get("turns", [[] for _ in range(n)])  # Per-sample turns
        # completion_masks: per-token mask (1=model, 0=env response) for multi-turn
        completion_masks_list = group.get("completion_masks", None)
        
        # Get total_tokens list from group if available, otherwise compute from token IDs
        total_tokens_list = group.get("total_tokens", [])
        
        for j in range(n):
            prompt_ids = group["prompt_token_ids"][j]
            comp_ids = group["completion_token_ids"][j]
            sample_metrics = sample_metrics_list[j] if j < len(sample_metrics_list) else {}
            golden_answers = golden_answers_list[j] if j < len(golden_answers_list) else {}
            vllm_logprobs = vllm_logprobs_list[j] if j < len(vllm_logprobs_list) else []
            turns = turns_list[j] if j < len(turns_list) else []
            
            # Get completion_mask if available (multi-turn interleaved)
            comp_mask = None
            if completion_masks_list is not None and j < len(completion_masks_list):
                comp_mask = completion_masks_list[j]
            
            # Use total_tokens from group if available, otherwise compute
            total_tokens = total_tokens_list[j] if j < len(total_tokens_list) else len(prompt_ids) + len(comp_ids)
            
            samples.append({
                "prompt_ids": prompt_ids,
                "comp_ids": comp_ids,
                "comp_mask": comp_mask,  # None for single-turn, list[int] for multi-turn
                "reward": group["rewards"][j],
                "advantage": group["advantages"][j],
                "sample_metrics": sample_metrics,  # Per-sample metrics (reward components + other metrics)
                "golden_answers": golden_answers,  # Dict mapping reward_name -> golden_answer
                "env_name": env_name,
                "prompt_text": prompt_text,
                "completion_text": completion_texts[j] if j < len(completion_texts) else "",
                "tokens_prompt": len(prompt_ids),
                "total_tokens": total_tokens,
                "vllm_logprobs": vllm_logprobs,
                "group_id": group_id,
                "turns": turns,
                "system_prompt": group.get("system_prompt", ""),
                "tokens_system_prompt": group.get("tokens_system_prompt", 0),
            })
    
    return samples



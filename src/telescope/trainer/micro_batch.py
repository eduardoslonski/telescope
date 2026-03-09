"""
Micro batching and sequence packing for efficient RL training.

This module implements:
1. Sequence packing - pack multiple samples into a single sequence to maximize GPU utilization
2. Dynamic micro batching - calculate number of micro batches based on total tokens
3. First Fit Decreasing bin packing algorithm for efficient packing

"""
from dataclasses import dataclass, field
import torch


@dataclass
class MicroBatch:
    """
    A single micro batch containing packed sequences.
    
    Multiple samples can be packed into a single micro batch.
    Position IDs reset for each sample to maintain proper attention.
    """
    input_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)  # 1 for completion tokens, 0 otherwise
    advantages: list[float] = field(default_factory=list)
    vllm_logprobs: list[float] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.input_ids)

    def can_fit(self, sample_len: int, max_seq_len: int) -> bool:
        """Check if a sample of given length can fit in this micro batch."""
        return len(self.input_ids) + sample_len <= max_seq_len

    def add_sample(
        self,
        input_ids: list[int],
        loss_mask: list[int],
        advantage: float,
        vllm_logprobs: list[float],
    ):
        """Add a sample to this micro batch."""
        # Position IDs reset for each sample (important for attention)
        position_ids = list(range(len(input_ids)))
        
        self.input_ids.extend(input_ids)
        self.loss_mask.extend(loss_mask)
        # Advantage is per-token (same value for all tokens in the sample)
        self.advantages.extend([advantage] * len(input_ids))
        self.vllm_logprobs.extend(vllm_logprobs)
        self.position_ids.extend(position_ids)


@dataclass
class PreparedSample:
    """A single sample prepared for packing."""
    input_ids: list[int]
    loss_mask: list[int]
    advantage: float
    vllm_logprobs: list[float]
    
    def __len__(self) -> int:
        return len(self.input_ids)


def prepare_sample(
    prompt_ids: list[int],
    completion_ids: list[int],
    advantage: float,
    vllm_logprobs: list[float],
    max_seq_len: int,
    completion_mask: list[int] | None = None,
) -> PreparedSample:
    """
    Prepare a single sample for packing.
    
    Args:
        prompt_ids: Token IDs for the prompt
        completion_ids: Token IDs for the completion (may include env response
            tokens for multi-turn)
        advantage: Advantage value for this sample
        vllm_logprobs: Log probabilities from vLLM for each token
        max_seq_len: Maximum sequence length (truncate if longer)
        completion_mask: Optional per-token mask for the completion.  When
            provided (multi-turn), 1 means model-generated token (train on)
            and 0 means env response token (mask out).  When None
            (single-turn), all completion tokens are treated as trainable.
    
    Returns:
        PreparedSample ready for packing
    """
    input_ids = prompt_ids + completion_ids
    
    # Build loss_mask: 0 for prompt, then use completion_mask if provided
    if completion_mask is not None:
        # Multi-turn: use the provided per-token mask
        loss_mask = [0] * len(prompt_ids) + list(completion_mask)
    else:
        # Single-turn: all completion tokens are trainable
        loss_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    
    # Pad vllm_logprobs to match sequence length
    # Prompt tokens get 0.0 logprob (not used in loss)
    n_comp = len(completion_ids)
    n_vllm = len(vllm_logprobs)
    
    if n_vllm < n_comp:
        vllm_logprobs = vllm_logprobs + [0.0] * (n_comp - n_vllm)
    elif n_vllm > n_comp:
        vllm_logprobs = vllm_logprobs[:n_comp]
    
    full_vllm_logprobs = [0.0] * len(prompt_ids) + vllm_logprobs
    
    # Truncate if too long
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        loss_mask = loss_mask[:max_seq_len]
        full_vllm_logprobs = full_vllm_logprobs[:max_seq_len]
    
    return PreparedSample(
        input_ids=input_ids,
        loss_mask=loss_mask,
        advantage=advantage,
        vllm_logprobs=full_vllm_logprobs,
    )


def pack_samples_into_micro_batches(
    samples: list[PreparedSample],
    max_seq_len: int,
) -> list[MicroBatch]:
    """
    Pack samples into micro batches using First Fit Decreasing algorithm.
    
    This minimizes padding by sorting samples by length (largest first)
    and fitting them into bins (micro batches) of max_seq_len tokens.
    
    Args:
        samples: List of prepared samples to pack
        max_seq_len: Maximum tokens per micro batch
    
    Returns:
        List of MicroBatch objects
    """
    # Sort samples by length (largest first) for better packing
    sorted_samples = sorted(samples, key=lambda x: -len(x))
    
    micro_batches: list[MicroBatch] = []
    
    for sample in sorted_samples:
        # Try to find a micro batch that can fit this sample
        placed = False
        for mb in micro_batches:
            if mb.can_fit(len(sample), max_seq_len):
                mb.add_sample(
                    input_ids=sample.input_ids,
                    loss_mask=sample.loss_mask,
                    advantage=sample.advantage,
                    vllm_logprobs=sample.vllm_logprobs,
                )
                placed = True
                break
        
        # If no existing micro batch can fit it, create a new one
        if not placed:
            new_mb = MicroBatch()
            new_mb.add_sample(
                input_ids=sample.input_ids,
                loss_mask=sample.loss_mask,
                advantage=sample.advantage,
                vllm_logprobs=sample.vllm_logprobs,
            )
            micro_batches.append(new_mb)
    
    return micro_batches


def pad_micro_batch(
    micro_batch: MicroBatch,
    pad_to_multiple_of: int,
    pad_token_id: int = 0,
) -> MicroBatch:
    """
    Pad a micro batch to a multiple of pad_to_multiple_of.
    
    Args:
        micro_batch: The micro batch to pad
        pad_to_multiple_of: Pad length to this multiple
        pad_token_id: Token ID to use for padding
    
    Returns:
        Padded micro batch
    """
    current_len = len(micro_batch)
    if pad_to_multiple_of <= 1:
        return micro_batch
    
    remainder = current_len % pad_to_multiple_of
    if remainder == 0:
        return micro_batch
    
    padding_size = pad_to_multiple_of - remainder
    
    micro_batch.input_ids.extend([pad_token_id] * padding_size)
    micro_batch.loss_mask.extend([0] * padding_size)
    micro_batch.advantages.extend([0.0] * padding_size)
    micro_batch.vllm_logprobs.extend([0.0] * padding_size)
    micro_batch.position_ids.extend(list(range(padding_size)))
    
    return micro_batch


def distribute_micro_batches(
    micro_batches: list[MicroBatch],
    num_ranks: int,
) -> list[list[MicroBatch]]:
    """
    Distribute micro batches across ranks with load balancing.

    Sorts micro batches by token count (descending) and uses zigzag
    assignment to balance total tokens per rank. This minimizes idle
    time in synchronized training (FSDP, Megatron) where all ranks
    must participate in each forward/backward pass.

    Ensures each rank gets the same number of micro batches by adding
    padding batches if necessary (with zero advantages so they don't
    contribute to loss).

    Args:
        micro_batches: List of all micro batches
        num_ranks: Number of trainer ranks

    Returns:
        List of micro batch lists, one per rank
    """
    # Work on a copy to avoid mutating the caller's list
    micro_batches = list(micro_batches)

    # Add padding batches if not evenly divisible
    num_padding = (-len(micro_batches)) % num_ranks
    if num_ranks > 1 and num_padding > 0 and micro_batches:
        # Create padding batch from first batch but with zero advantages
        template = micro_batches[0]
        for _ in range(num_padding):
            pad_batch = MicroBatch(
                input_ids=template.input_ids.copy(),
                loss_mask=[0] * len(template.loss_mask),  # All masked out
                advantages=[0.0] * len(template.advantages),
                vllm_logprobs=template.vllm_logprobs.copy(),
                position_ids=template.position_ids.copy(),
            )
            micro_batches.append(pad_batch)

    # Sort by token count (descending) and zigzag-assign across ranks so that
    # each rank gets a balanced mix of large and small micro batches.  This
    # minimizes idle time in synchronized training (FSDP all-gather/reduce-
    # scatter, Megatron TP/PP) where the slowest rank gates wall-clock time.
    #
    # Zigzag pattern for 4 ranks:
    #   row 0 (largest):  R0  R1  R2  R3
    #   row 1:            R3  R2  R1  R0
    #   row 2:            R0  R1  R2  R3
    #   ...
    micro_batches.sort(key=lambda mb: -len(mb))

    batches_per_rank: list[list[MicroBatch]] = [[] for _ in range(num_ranks)]

    for i, mb in enumerate(micro_batches):
        row = i // num_ranks
        pos = i % num_ranks
        rank = pos if row % 2 == 0 else num_ranks - 1 - pos
        batches_per_rank[rank].append(mb)

    return batches_per_rank


def micro_batches_to_tensors(micro_batches: list[MicroBatch]) -> list[dict]:
    """
    Convert list of MicroBatch objects to list of tensor dictionaries.
    
    Each micro batch becomes a dict with tensors shaped [1, seq_len].
    
    Args:
        micro_batches: List of MicroBatch objects
    
    Returns:
        List of dicts with tensor data
    """
    result = []
    for mb in micro_batches:
        result.append({
            "input_ids": torch.tensor([mb.input_ids], dtype=torch.long),
            "loss_mask": torch.tensor([mb.loss_mask], dtype=torch.bool),
            "advantages": torch.tensor([mb.advantages], dtype=torch.float),
            "vllm_logprobs": torch.tensor([mb.vllm_logprobs], dtype=torch.float),
            "position_ids": torch.tensor([mb.position_ids], dtype=torch.long),
        })
    return result


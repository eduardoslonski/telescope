"""Context parallelism utilities for FSDP ring attention.

Provides helpers to shard sequences across CP ranks and set up the
ring-flash-attn kernel parameters.  Uses contiguous sequence chunking
with CP folded into the FSDP shard dimension.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def shard_for_cp(
    t: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
) -> torch.Tensor:
    """Split *t* along dim-1 (sequence) and return this rank's chunk.

    Args:
        t: Tensor of shape ``[1, seq_len, ...]``.
        cp_rank: This rank's position inside the CP group.
        cp_world_size: Total number of CP ranks.

    Returns:
        Contiguous chunk ``[1, seq_len // cp_world_size, ...]``.
    """
    return torch.chunk(t, cp_world_size, dim=1)[cp_rank].contiguous()


def _get_cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Derive cumulative sequence lengths from packed ``position_ids``.

    In packed sequences, ``position_ids`` resets to 0 at each sample boundary.
    This function finds those boundaries and returns ``cu_seqlens`` in the
    format expected by ``ring_flash_attn.update_ring_flash_attn_params``.
    """
    flat = position_ids.view(-1)
    seqlens = torch.cat([
        flat[0:1],
        flat[:-1][(flat == 0)[1:]] + 1,
        flat[-1:] + 1,
    ])
    return seqlens.cumsum(dim=0, dtype=torch.int32)


def setup_cp_params(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
    cp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare inputs for a ring-attention forward pass.

    1. Compute ``cu_seqlens`` from the *full* ``position_ids`` (before sharding)
       and register them with the ring-flash-attn kernel.
    2. Shard both ``input_ids`` and ``position_ids`` for this CP rank.

    Returns:
        ``(sharded_input_ids, sharded_position_ids)``
    """
    from ring_flash_attn import update_ring_flash_attn_params

    # cu_seqlens must be computed on the full (un-sharded) position_ids so the
    # ring kernel knows the original packed-sample boundaries.
    cu_seqlens = _get_cu_seqlens_from_position_ids(position_ids)
    update_ring_flash_attn_params(cu_seqlens, cp_group)

    input_ids = shard_for_cp(input_ids, cp_rank, cp_world_size)
    position_ids = shard_for_cp(position_ids, cp_rank, cp_world_size)
    return input_ids, position_ids

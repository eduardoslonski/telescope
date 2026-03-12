"""Weight synchronization between trainer and inference servers."""
from __future__ import annotations

import pickle
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from telescope.utils import config
from telescope.utils.tlog import get_logger

if TYPE_CHECKING:
    from telescope.trainer.backends import TrainingBackend
    from telescope.trainer.metrics.timeline import GPUTimelineLogger, _NullTracker

_log = get_logger("trainer")

# Module-level caches – model architecture is fixed during training, so metadata
# (tensor names, shapes, dtypes, bucket layout) only needs to be sent once per
# communicator.  Keyed by communicator object id.
_metadata_sent_comms: set[int] = set()
_flattened_chunks_cache: dict[int, list[dict]] = {}


def _dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _name_to_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype name in broadcast metadata: {dtype_name}")
    return getattr(torch, dtype_name)


def _stage_state_dict_to_cpu(state_dict: dict, rank: int) -> dict:
    """
    Move all tensor values in a state_dict to CPU.

    This is used to reduce GPU peak memory before NCCL broadcast by keeping only
    one tensor on GPU at a time during send.

    GPU tensors are freed incrementally via ``.pop()`` so that at most one
    parameter's GPU + CPU copies coexist at any time, rather than keeping the
    entire GPU state dict alive until the caller drops the old reference.
    """
    staged = {}
    moved_count = 0
    keys = list(state_dict.keys())
    for key in keys:
        value = state_dict.pop(key)
        if isinstance(value, torch.Tensor) and value.is_cuda:
            staged[key] = value.detach().cpu()
            del value
            moved_count += 1
        else:
            staged[key] = value

    if moved_count > 0:
        _log.info(
            f"Staged {moved_count} tensors to CPU before broadcast",
            rank=rank,
        )
        torch.cuda.empty_cache()
    return staged


def _build_flattened_bucket_chunks(
    state_dict: dict,
    max_bucket_bytes: int,
) -> list[dict]:
    """Build flattened bucket chunk metadata from a state dict."""
    if max_bucket_bytes <= 0:
        raise ValueError(f"max_bucket_bytes must be > 0, got {max_bucket_bytes}")

    chunks: list[dict] = []
    pending: list[tuple[str, torch.Tensor]] = []
    pending_bytes = 0

    def _flush_pending() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return

        # Keep deterministic order while grouping by dtype.
        grouped: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for key, tensor in pending:
            dtype_name = _dtype_to_name(tensor.dtype)
            grouped.setdefault(dtype_name, []).append((key, tensor))

        for dtype_name, entries in grouped.items():
            keys = [key for key, _ in entries]
            shapes = [tuple(tensor.shape) for _, tensor in entries]
            numels = [int(tensor.numel()) for _, tensor in entries]
            chunks.append(
                {
                    "dtype": dtype_name,
                    "keys": keys,
                    "shapes": shapes,
                    "numels": numels,
                }
            )

        pending = []
        pending_bytes = 0

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(value)!r}")

        tensor_bytes = int(value.numel() * value.element_size())
        if pending and pending_bytes + tensor_bytes > max_bucket_bytes:
            _flush_pending()
        pending.append((key, value))
        pending_bytes += tensor_bytes

    _flush_pending()
    return chunks


def _broadcast_metadata(metadata: object, communicator: PyNcclCommunicator, rank: int) -> None:
    """Broadcast arbitrary pickled metadata blob.

    Model architecture is fixed during training, so metadata (tensor names,
    shapes, dtypes) is identical on every broadcast.  On the first call per
    communicator we send the full payload; on subsequent calls we send
    metadata_size=0 which tells the receiver to reuse its cached copy.
    """
    comm_id = id(communicator)
    if comm_id in _metadata_sent_comms:
        tensor_metadata_size = torch.tensor(
            [0], dtype=torch.int64, device=communicator.device,
        )
        communicator.broadcast(tensor_metadata_size, src=0)
        _log.debug("Skipped metadata broadcast (cached on receiver)", rank=rank)
        return

    metadata_bytes = pickle.dumps(metadata)
    tensor_metadata_size = torch.tensor(
        [len(metadata_bytes)],
        dtype=torch.int64,
        device=communicator.device,
    )
    _log.debug("Sending metadata size to inference", rank=rank)
    communicator.broadcast(tensor_metadata_size, src=0)
    _log.debug("Metadata size sent", rank=rank)

    tensor_metadata = torch.ByteTensor(list(metadata_bytes)).to(communicator.device)
    _log.debug("Sending metadata blob", rank=rank)
    communicator.broadcast(tensor_metadata, src=0)
    _log.debug("Metadata sent and cached for future broadcasts", rank=rank)
    _metadata_sent_comms.add(comm_id)


def _broadcast_state_dict_flattened_buckets(
    state_dict: dict,
    communicator: PyNcclCommunicator,
    rank: int,
) -> None:
    """Broadcast state dict via flattened buckets."""
    comm_id = id(communicator)
    cached_chunks = _flattened_chunks_cache.get(comm_id)
    if cached_chunks is not None:
        chunks = cached_chunks
    else:
        bucket_mb = int(config.cfg.weight_broadcast_bucket_mb)
        max_bucket_bytes = max(1, bucket_mb) * 1024 * 1024
        chunks = _build_flattened_bucket_chunks(state_dict, max_bucket_bytes=max_bucket_bytes)
        _flattened_chunks_cache[comm_id] = chunks

    metadata = {
        "format": "flattened_bucket_v1",
        "chunks": chunks,
    }
    _broadcast_metadata(metadata, communicator, rank)

    for chunk in chunks:
        dtype = _name_to_dtype(chunk["dtype"])
        keys = chunk["keys"]
        numels = chunk["numels"]
        total_numel = int(sum(numels))
        flat_tensor = torch.empty(total_numel, dtype=dtype, device=communicator.device)

        offset = 0
        for key, numel in zip(keys, numels):
            source_tensor = state_dict[key]
            if not isinstance(source_tensor, torch.Tensor):
                raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(source_tensor)!r}")

            if not source_tensor.is_cuda:
                # CPU staging path: copy directly CPU->GPU slice to avoid an extra
                # temporary GPU tensor allocation per parameter.
                flat_tensor[offset : offset + numel].copy_(source_tensor.view(-1))
            else:
                if not source_tensor.is_contiguous():
                    source_tensor = source_tensor.contiguous()
                flat_tensor[offset : offset + numel].copy_(source_tensor.view(-1))
            offset += numel

        communicator.broadcast(flat_tensor, src=0)
        del flat_tensor

    _log.debug(
        f"Sent {len(state_dict)} tensors in {len(chunks)} flattened buckets",
        rank=rank,
    )


def setup_inference_communicator(
    rank: int,
    device,
    master_address: str | None = None,
    master_port: int | None = None,
    num_inference_servers: int | None = None,
) -> PyNcclCommunicator | None:
    """
    Set up NCCL communicator with inference servers.
    
    Only the broadcast rank (rank 0 / tp_rank 0) needs this communicator.
    Returns None for other ranks.
    """
    if rank != 0:
        return None

    _log.info("Starting NCCL with inference", rank=rank)
    
    if num_inference_servers is None:
        total_inference_workers = int(config.cfg.inference_num_workers)
        if total_inference_workers < 1:
            raise ValueError(
                "INFERENCE_NUM_WORKERS must be set in `telescope.config` and be >= 1."
            )
        # Each inference GPU worker is an NCCL rank in the broadcast group
        num_inference_servers = total_inference_workers
    if master_address is None:
        master_address = "127.0.0.1"
    if master_port is None:
        master_port = config.cfg.inference_base_port + num_inference_servers
    world_size = 1 + num_inference_servers

    _log.info(f"Inference NCCL world_size={world_size} (1 trainer + {num_inference_servers} inference servers)", rank=rank)

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
        store_timeout=300,
    )

    _log.info("Starting communicator with inference ranks", rank=rank)
    communicator = PyNcclCommunicator(pg, device=device)
    _log.info("Communicator started", rank=rank)
    
    return communicator


def setup_inference_communicator_for_group(
    rank: int,
    device,
    master_address: str,
    master_port: int,
    num_servers_in_group: int,
    group_name: str = "training_only",
) -> PyNcclCommunicator | None:
    """
    Set up an additional NCCL communicator for a subset of inference servers.

    Used to create the *training-only* group that excludes eval servers.
    Returns None for non-broadcast ranks.
    """
    if rank != 0:
        return None

    world_size = 1 + num_servers_in_group
    _log.info(
        f"Inference NCCL group={group_name} world_size={world_size} "
        f"(1 trainer + {num_servers_in_group} servers)",
        rank=rank,
    )

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=0,
        world_size=world_size,
        store_timeout=300,
    )

    communicator = PyNcclCommunicator(pg, device=device)
    _log.info(f"Communicator started for group={group_name}", rank=rank)
    return communicator


def prepare_weights_for_broadcast(
    backend: TrainingBackend,
    step: int | None = None,
    tracker: GPUTimelineLogger | _NullTracker | None = None,
) -> dict[str, torch.Tensor] | None:
    """
    Phase 1 of weight sync: gather weights and barrier.

    All trainer ranks must call this collectively.  Returns the gathered
    state dict on the broadcast rank (optionally CPU-staged), ``None``
    on other ranks.

    IMPORTANT: We must barrier AFTER gathering but BEFORE broadcasting.

    The gather operation uses collective ops (all-gather for FSDP shards or
    TP shards). If the broadcast rank finishes gathering and immediately starts
    broadcasting to inference (different NCCL group), other ranks get stuck
    waiting to participate in the gather = DEADLOCK.
    """
    rank = backend.rank
    is_broadcast_rank = backend.is_weight_broadcast_rank

    def _track(name: str):
        if tracker is not None:
            return tracker.track(name)
        return nullcontext()

    # Step 1: All ranks gather weights (collective operation)
    with _track("gather_state_dict"):
        _log.debug("Gathering state_dict for broadcast to inference", step=step, rank=rank)
        state_dict = backend.gather_weights_for_inference()

    # Step 2: Barrier after gathering, before broadcasting
    _log.debug("State dict gathered; waiting for all ranks before broadcast", step=step, rank=rank)
    with _track("barrier_pre"):
        backend.barrier()
    _log.debug("Barrier before broadcast completed", step=step, rank=rank)

    # Non-broadcast ranks no longer need gathered weights.
    if not is_broadcast_rank:
        del state_dict
        return None

    # Optional: stage to CPU to reduce GPU peak memory during broadcast.
    if config.cfg.weight_broadcast_cpu_staging:
        with _track("stage_state_dict_cpu"):
            state_dict = _stage_state_dict_to_cpu(state_dict, rank)

    return state_dict


def broadcast_weights_to_inference(
    backend: TrainingBackend,
    state_dict: dict[str, torch.Tensor] | None,
    communicator: PyNcclCommunicator | None,
    step: int | None = None,
    tracker: GPUTimelineLogger | _NullTracker | None = None,
):
    """
    Phase 2 of weight sync: NCCL broadcast and final barrier.

    All trainer ranks must call this collectively.  Only the broadcast rank
    actually sends data; other ranks participate in the final barrier.
    """
    rank = backend.rank
    is_broadcast_rank = backend.is_weight_broadcast_rank

    def _track(name: str):
        if tracker is not None:
            return tracker.track(name)
        return nullcontext()

    should_cleanup_state_dict = False
    if is_broadcast_rank:
        _log.debug("Starting broadcast to inference", step=step, rank=rank)
        with _track("nccl_broadcast"):
            _broadcast_state_dict(state_dict, communicator, rank)
        should_cleanup_state_dict = True
        _log.debug("Broadcast to inference finished", step=step, rank=rank)
    else:
        _log.debug("Waiting for broadcast rank to finish sending to inference", step=step, rank=rank)

    # Final sync barrier
    with _track("barrier_post"):
        backend.barrier()
    _log.debug("Barrier after broadcast completed", step=step, rank=rank)

    # Cleanup after barrier so non-broadcast ranks do not wait on rank0's Python GC
    # while sitting inside barrier_post.
    if should_cleanup_state_dict:
        with _track("cleanup_state_dict"):
            del state_dict


def send_weights_to_inference(
    backend: TrainingBackend,
    communicator: PyNcclCommunicator | None,
    step: int | None = None,
    tracker: GPUTimelineLogger | _NullTracker | None = None,
):
    """
    Send model weights from trainer to inference via NCCL broadcast.

    Convenience wrapper that calls :func:`prepare_weights_for_broadcast`
    followed by :func:`broadcast_weights_to_inference`.
    """
    state_dict = prepare_weights_for_broadcast(backend, step=step, tracker=tracker)
    broadcast_weights_to_inference(backend, state_dict, communicator, step=step, tracker=tracker)


def _broadcast_state_dict(state_dict: dict, communicator: PyNcclCommunicator, rank: int):
    """Broadcast state dict to inference servers via NCCL."""
    if communicator is None:
        raise RuntimeError("Communicator not initialized for broadcast rank")

    broadcast_mode = str(config.cfg.weight_broadcast_mode).strip().lower()
    if broadcast_mode == "flattened_bucket":
        _broadcast_state_dict_flattened_buckets(state_dict, communicator, rank)
        return
    if broadcast_mode != "per_tensor":
        raise ValueError(
            f"Unsupported WEIGHT_BROADCAST_MODE={broadcast_mode!r}. "
            "Expected 'flattened_bucket' or 'per_tensor'."
        )

    # Send metadata first
    metadata = {key: (value.shape, value.dtype) for key, value in state_dict.items()}
    _broadcast_metadata(metadata, communicator, rank)

    # Send all tensors
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(tensor)!r}")

        if not tensor.is_cuda:
            # CPU staging mode: move one tensor at a time to limit GPU peak memory.
            send_tensor = tensor.to(communicator.device)
        else:
            send_tensor = tensor

        communicator.broadcast(send_tensor, src=0)

    _log.debug(f"Sent {len(state_dict)} tensors in state_dict", rank=rank)

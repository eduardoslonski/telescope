"""
Worker extension for vLLM that handles weight updates via NCCL.

This worker receives model weights from the trainer process using NCCL
broadcast, allowing in-flight weight updates during inference.
"""
import pickle
import time
import os

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.utils import StatelessProcessGroup

from telescope.utils.tlog import get_logger, setup_logging

# Module-level logger (initialized lazily)
_logger = None


def _get_logger():
    """Get or initialize the inference logger."""
    global _logger
    if _logger is None:
        setup_logging()
        _logger = get_logger("inference")
    return _logger


class NCCLWeightUpdateWorker:
    """vLLM worker extension that receives weights from trainer via NCCL.

    Supports dual NCCL groups for eval mode:
    - ``"full"``: trainer + all inference servers (default)
    - ``"training_only"``: trainer + non-eval inference servers
    Eval-only servers only join the full group.
    """

    # Note: __init__ is NOT called by vLLM worker extensions - they're mixed in.
    # Use lazy initialization in methods instead.

    def _ensure_communicators_dict(self):
        if not hasattr(self, "_communicators"):
            self._communicators: dict[str, PyNcclCommunicator] = {}

    def _log(self, message: str):
        """Log with rank prefix."""
        tp_rank = getattr(self, "_tp_rank", None)
        if tp_rank is None:
            try:
                tp_rank = get_tp_group().rank
                self._tp_rank = tp_rank
            except Exception:
                tp_rank = -1
        _get_logger().info(message, rank=tp_rank)

    def collective_test(self):
        """Test endpoint to verify worker is accessible."""
        self._log("collective_test called")
        self._log(f"model_runner={self.model_runner}")

    def init_broadcast(self, host: str, port: int, world_size: int, rank: int = None, group: str = "full"):
        """
        Initialize NCCL communicator for weight updates.

        Args:
            host: Master address
            port: Master port
            world_size: Total ranks (trainer + all inference TP workers in this group)
            rank: Base rank for this inference server (each TP worker offsets by its TP rank)
            group: Group name (``"full"`` or ``"training_only"``)
        """
        tp_rank = get_tp_group().rank
        if rank is None:
            nccl_rank = tp_rank + 1
        else:
            nccl_rank = rank + tp_rank
        self._tp_rank = tp_rank
        rank = nccl_rank
        self._ensure_communicators_dict()

        self._log(f"init_broadcast group={group} host={host} port={port} world_size={world_size} rank={rank}")
        comm = self._build_communicator(host, port, rank, world_size)
        self._communicators[group] = comm
        self._log(f"communicator initialized for group={group}")

    def _build_communicator(self, host: str, port: int, rank: int, world_size: int) -> PyNcclCommunicator:
        """Build the NCCL communicator."""
        self._log("building StatelessProcessGroup")
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
            store_timeout=300,
        )
        self._log("starting NCCL communicator")
        return PyNcclCommunicator(pg, device=self.device)

    def _get_communicator(self, group: str = "full") -> PyNcclCommunicator:
        self._ensure_communicators_dict()
        comm = self._communicators.get(group)
        if comm is None:
            raise RuntimeError(f"No communicator for group={group!r}. Available: {list(self._communicators)}")
        return comm

    def receive_state_dict(self, group: str = "full"):
        """
        Receive state dict from trainer via NCCL broadcast.

        Yields (key, tensor) pairs as they are received.
        """
        comm = self._get_communicator(group)
        recv_start = time.time()
        self._log(f"receiving state_dict from trainer (group={group})")

        metadata = self._receive_metadata(comm)
        tensor_count = 0

        if isinstance(metadata, dict) and metadata.get("format") == "flattened_bucket_v1":
            chunks = metadata.get("chunks", [])
            for chunk in chunks:
                dtype = self._metadata_dtype_to_torch_dtype(chunk["dtype"])
                keys = chunk["keys"]
                shapes = chunk["shapes"]
                numels = chunk["numels"]

                total_numel = int(sum(numels))
                flat_tensor = torch.empty(total_numel, dtype=dtype, device=comm.device)
                comm.broadcast(flat_tensor, src=0)

                offset = 0
                for key, shape, numel in zip(keys, shapes, numels):
                    view = flat_tensor[offset : offset + int(numel)].view(tuple(shape))
                    offset += int(numel)
                    tensor_count += 1
                    yield key, view
        else:
            for key, value in metadata.items():
                tensor = self._receive_tensor(value, comm)
                tensor_count += 1
                yield key, tensor

        elapsed = time.time() - recv_start
        self._log(f"received {tensor_count} tensors in {elapsed:.2f}s (group={group})")

    def _receive_metadata(self, comm: PyNcclCommunicator) -> dict:
        """Receive metadata describing the state dict.

        The trainer sends metadata_size=0 after the first broadcast to signal
        that the model architecture hasn't changed and the receiver should
        reuse its cached copy.
        """
        start = time.time()

        tensor_metadata_size = torch.tensor([0], dtype=torch.int64, device=comm.device)
        self._log("waiting for metadata size")
        comm.broadcast(tensor_metadata_size, src=0)
        metadata_size = tensor_metadata_size.item()

        if metadata_size == 0:
            cache = getattr(self, "_metadata_cache", {})
            cached = cache.get(id(comm))
            if cached is None:
                raise RuntimeError(
                    "Trainer signalled cached metadata (size=0) but no "
                    "cached metadata available for this communicator"
                )
            self._log(f"using cached metadata ({time.time() - start:.3f}s)")
            return cached

        self._log(f"received metadata size {metadata_size} ({time.time() - start:.3f}s)")

        metadata_bytes = torch.empty(metadata_size, dtype=torch.uint8).to(comm.device)
        meta_start = time.time()
        self._log("waiting for metadata bytes")
        comm.broadcast(metadata_bytes, src=0)
        self._log(f"received metadata bytes in {time.time() - meta_start:.3f}s")

        metadata = pickle.loads(bytes(metadata_bytes.cpu().numpy()))

        if not hasattr(self, "_metadata_cache"):
            self._metadata_cache: dict[int, dict] = {}
        self._metadata_cache[id(comm)] = metadata
        self._log(f"metadata received and cached for future broadcasts ({time.time() - start:.3f}s)")

        return metadata

    def _receive_tensor(self, metadata_entry: tuple, comm: PyNcclCommunicator) -> torch.Tensor:
        """Receive a single tensor from the broadcast."""
        shape, dtype = metadata_entry
        if isinstance(dtype, str):
            dtype = self._metadata_dtype_to_torch_dtype(dtype)
        tensor = torch.empty(shape, dtype=dtype, device=comm.device)
        comm.broadcast(tensor, src=0)
        return tensor

    @staticmethod
    def _metadata_dtype_to_torch_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name.split(".", 1)[1]
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported dtype in weight metadata: {dtype_name}")
        return getattr(torch, dtype_name)

    def load_weights(self, group: str = "full"):
        """Load new weights from trainer using the specified NCCL group."""
        start = time.time()
        self._log(f"loading weights from trainer (group={group})")

        model = self.model_runner.model
        state_iter = self.receive_state_dict(group=group)
        model.load_weights(state_iter)

        self._log(f"weights loaded in {time.time() - start:.2f}s (group={group})")

    def collect_torch_memory_metrics(self) -> list[dict]:
        """
        Return current torch CUDA allocator metrics for this vLLM worker process.

        This runs inside the inference worker process (where model tensors live),
        so values reflect allocator state for the actual serving process.
        """
        try:
            device_obj = getattr(self, "device", None)
            if isinstance(device_obj, torch.device):
                local_cuda_index = (
                    int(device_obj.index)
                    if device_obj.index is not None
                    else int(torch.cuda.current_device())
                )
            elif isinstance(device_obj, int):
                local_cuda_index = int(device_obj)
            else:
                local_cuda_index = int(torch.cuda.current_device())
            device = torch.device(f"cuda:{local_cuda_index}")
        except Exception:
            return []

        try:
            allocated = float(torch.cuda.memory_allocated(device) / (1024 ** 3))
            reserved = float(torch.cuda.memory_reserved(device) / (1024 ** 3))
            max_allocated = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        except Exception:
            return []

        physical_gpu_index = local_cuda_index
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            try:
                visible_devices = [int(part.strip()) for part in cuda_visible.split(",")]
                if 0 <= local_cuda_index < len(visible_devices):
                    physical_gpu_index = int(visible_devices[local_cuda_index])
            except Exception:
                pass

        try:
            tp_rank = int(get_tp_group().rank)
        except Exception:
            tp_rank = -1

        timestamp = time.time()
        return [
            {
                "timestamp": timestamp,
                "gpu_index": physical_gpu_index,
                "metric_name": "torch_allocated_gb",
                "value": allocated,
                "source": "torch_inference",
                "tp_rank": tp_rank,
                "local_rank": local_cuda_index,
                "rank": -1,
                "node_id": -1,
            },
            {
                "timestamp": timestamp,
                "gpu_index": physical_gpu_index,
                "metric_name": "torch_reserved_gb",
                "value": reserved,
                "source": "torch_inference",
                "tp_rank": tp_rank,
                "local_rank": local_cuda_index,
                "rank": -1,
                "node_id": -1,
            },
            {
                "timestamp": timestamp,
                "gpu_index": physical_gpu_index,
                "metric_name": "torch_max_allocated_gb",
                "value": max_allocated,
                "source": "torch_inference",
                "tp_rank": tp_rank,
                "local_rank": local_cuda_index,
                "rank": -1,
                "node_id": -1,
            },
        ]


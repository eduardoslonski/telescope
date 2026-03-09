"""
PyTorch GPU memory logger for trainer processes.

Runs a background thread that samples torch.cuda memory at a configurable
interval and buffers samples in memory for the orchestrator to drain via Ray.

This captures memory metrics from the trainer's perspective:
- allocated_gb: Memory currently allocated by PyTorch
- reserved_gb: Memory reserved by the CUDA memory allocator
- max_allocated_gb: Peak memory allocated since last reset

Each trainer rank has its own logger.
"""
import os
import threading
import time
from dataclasses import dataclass

import torch

from telescope.utils.tlog import get_logger

_log = get_logger("trainer")


@dataclass
class TorchMemorySample:
    """Single memory sample."""
    timestamp: float
    gpu_index: int  # Physical GPU index
    metric_name: str
    value: float
    rank: int
    local_rank: int
    node_id: int
    source: str = "torch_trainer"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "gpu_index": self.gpu_index,
            "metric_name": self.metric_name,
            "value": self.value,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "node_id": self.node_id,
            "source": self.source,
        }


class TorchMemoryLogger:
    """
    Background thread that samples PyTorch GPU memory at a configurable interval
    and buffers samples in memory for remote draining via Ray.

    Usage:
        logger = TorchMemoryLogger(rank=0, local_rank=0, node_id=0)
        logger.start()
        # ... training ...
        samples = logger.drain_samples()
        logger.stop()
    """

    DEFAULT_SAMPLE_INTERVAL_MS = 100

    def __init__(
        self,
        rank: int,
        local_rank: int,
        node_id: int | str,
        sample_interval_ms: int | float = DEFAULT_SAMPLE_INTERVAL_MS,
    ):
        self.rank = rank
        self.local_rank = local_rank
        try:
            self.node_id = int(node_id)
        except (TypeError, ValueError):
            self.node_id = -1
        self.device = torch.device(f"cuda:{local_rank}")
        self.sample_interval_ms = self._normalize_sample_interval_ms(sample_interval_ms)

        # Determine physical GPU index
        self.physical_gpu_index = self._get_physical_gpu_index()

        # State
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples_lock = threading.Lock()
        self._pending_samples: list[TorchMemorySample] = []

    @classmethod
    def _normalize_sample_interval_ms(cls, raw_value: int | float) -> int:
        """Validate sampling interval (milliseconds), falling back to default."""
        try:
            interval_ms = int(raw_value)
        except (TypeError, ValueError):
            _log.warning(
                f"Invalid torch memory sampling interval {raw_value!r}; "
                f"using default {cls.DEFAULT_SAMPLE_INTERVAL_MS}ms."
            )
            return cls.DEFAULT_SAMPLE_INTERVAL_MS

        if interval_ms < 1:
            _log.warning(
                f"Torch memory sampling interval must be >= 1ms, got {interval_ms}; "
                f"using default {cls.DEFAULT_SAMPLE_INTERVAL_MS}ms."
            )
            return cls.DEFAULT_SAMPLE_INTERVAL_MS

        return interval_ms

    def _get_physical_gpu_index(self) -> int:
        """
        Map local_rank to physical GPU index using CUDA_VISIBLE_DEVICES.

        If CUDA_VISIBLE_DEVICES is set, use it to map local_rank to physical GPU.
        Otherwise, assume local_rank == physical GPU index.
        """
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            devices = [int(d.strip()) for d in cuda_visible.split(",")]
            if self.local_rank < len(devices):
                return devices[self.local_rank]
        return self.local_rank

    def _sample_memory(self) -> list[TorchMemorySample]:
        """Sample current GPU memory state."""
        timestamp = time.time()
        samples = []

        try:
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

            samples.append(TorchMemorySample(
                timestamp=timestamp,
                gpu_index=self.physical_gpu_index,
                metric_name="torch_allocated_gb",
                value=allocated,
                rank=self.rank,
                local_rank=self.local_rank,
                node_id=self.node_id,
            ))
            samples.append(TorchMemorySample(
                timestamp=timestamp,
                gpu_index=self.physical_gpu_index,
                metric_name="torch_reserved_gb",
                value=reserved,
                rank=self.rank,
                local_rank=self.local_rank,
                node_id=self.node_id,
            ))
            samples.append(TorchMemorySample(
                timestamp=timestamp,
                gpu_index=self.physical_gpu_index,
                metric_name="torch_max_allocated_gb",
                value=max_allocated,
                rank=self.rank,
                local_rank=self.local_rank,
                node_id=self.node_id,
            ))
        except Exception:
            pass

        return samples

    def _collection_loop(self):
        """Background loop that samples memory at the configured interval."""
        sample_interval_sec = self.sample_interval_ms / 1000.0

        while not self._stop_event.is_set():
            new_samples = self._sample_memory()
            if new_samples:
                with self._samples_lock:
                    self._pending_samples.extend(new_samples)
            self._stop_event.wait(timeout=sample_interval_sec)

    def start(self):
        """Start the background sampling thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        _log.debug(f"TorchMemoryLogger started for GPU {self.physical_gpu_index}", rank=self.rank)

    def stop(self):
        """Stop the background sampling thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        _log.debug("TorchMemoryLogger stopped", rank=self.rank)

    def drain_samples(self) -> list[dict]:
        """
        Drain in-memory samples collected since the last call.

        Returns:
            List of serialized TorchMemorySample dictionaries.
        """
        with self._samples_lock:
            if not self._pending_samples:
                return []
            samples = [s.to_dict() for s in self._pending_samples]
            self._pending_samples.clear()
        return samples

"""
System metrics logging for real-time hardware monitoring.

This module collects system metrics (GPU, CPU, memory) and provides them
to EventLogger for unified upload to W&B. Data is collected in the events folder:
- events/tail.zip: Contains gpu.parquet + cpu.parquet (along with other event parquet files)
- events/block_live.zip: Current 30-minute block
- events/block_*.zip: Finalized 30-minute blocks

Metrics are collected at a configurable interval (1 second by default).

GPU metrics (per GPU, with gpu_index column):
- gpu_memory_used_gb, gpu_memory_free_gb, gpu_memory_used_percent
- gpu_temperature_c
- gpu_power_w
- gpu_utilization_percent, gpu_memory_bandwidth_utilization_percent
- gpu_clock_sm_mhz, gpu_clock_mem_mhz
- gpu_fan_speed_percent

CPU/System metrics:
- cpu_utilization_percent (overall), cpu_core_{i}_utilization_percent (per-core)
- system_memory_used_gb, system_memory_available_gb, system_memory_percent
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from dataclasses import dataclass

from telescope.utils.tlog import get_logger

_log = get_logger("orchestrator")


@dataclass
class GpuMetricSample:
    """Single GPU metric sample."""
    timestamp: float
    gpu_index: int
    metric_name: str
    value: float
    node_id: int = -1
    node_ip: str = ""
    hostname: str = ""
    ray_node_id: str = ""
    rank: int = -1
    local_rank: int = -1
    source: str = "system"


@dataclass
class CpuMetricSample:
    """Single CPU/system metric sample."""
    timestamp: float
    metric_name: str
    value: float
    node_id: int = -1
    node_ip: str = ""
    hostname: str = ""
    ray_node_id: str = ""
    source: str = "system"


class SystemMetricsLogger:
    """
    Thread-safe system metrics logger that collects hardware metrics at a
    configurable interval and provides data to EventLogger for unified upload.
    
    Usage:
        logger = SystemMetricsLogger()
        logger.initialize(wandb_run)
        
        # Start collection loop
        await logger.start()
        
        # Get metrics for upload (called by EventLogger)
        gpu_metrics, cpu_metrics = logger.get_and_clear_metrics()
        
        # Stop when done
        await logger.stop()
    """

    DEFAULT_COLLECTION_INTERVAL_SECONDS = 1.0

    def __init__(self, collection_interval_seconds: float = DEFAULT_COLLECTION_INTERVAL_SECONDS):
        self.run = None
        self._lock = threading.Lock()
        self.node_id: int = -1
        self.node_ip: str = ""
        self.hostname: str = socket.gethostname()
        self.ray_node_id: str = ""
        self.collection_interval_seconds = self._normalize_collection_interval_seconds(
            collection_interval_seconds
        )

        # Separate buffers for GPU and CPU metrics
        self._gpu_metrics: list[GpuMetricSample] = []
        self._cpu_metrics: list[CpuMetricSample] = []

        # Background tasks
        self._stop_event: asyncio.Event | None = None
        self._collection_task: asyncio.Task | None = None

        # Reference time
        self._run_start_time: float = time.time()

        # GPU handles (initialized on first collection)
        self._nvml_initialized: bool = False
        self._gpu_handles: list = []
        self._num_gpus: int = 0

    @classmethod
    def _normalize_collection_interval_seconds(cls, raw_value: float) -> float:
        """Validate collection interval (seconds), falling back to default."""
        try:
            interval_seconds = float(raw_value)
        except (TypeError, ValueError):
            _log.warning(
                f"Invalid system metrics interval {raw_value!r}; using default "
                f"{cls.DEFAULT_COLLECTION_INTERVAL_SECONDS}s."
            )
            return cls.DEFAULT_COLLECTION_INTERVAL_SECONDS

        if interval_seconds <= 0:
            _log.warning(
                f"System metrics interval must be > 0, got {interval_seconds}; using "
                f"default {cls.DEFAULT_COLLECTION_INTERVAL_SECONDS}s."
            )
            return cls.DEFAULT_COLLECTION_INTERVAL_SECONDS

        return interval_seconds

    def initialize(
        self,
        wandb_run,
        node_id: int | None = None,
        node_ip: str | None = None,
        hostname: str | None = None,
        ray_node_id: str | None = None,
    ):
        """Initialize with a W&B run object."""
        self.run = wandb_run
        self._run_start_time = time.time()
        self.set_node_identity(
            node_id=node_id,
            node_ip=node_ip,
            hostname=hostname,
            ray_node_id=ray_node_id,
        )
        self._init_nvml()
        _log.debug(f"SystemMetricsLogger initialized with run {wandb_run.name if wandb_run else 'None'}")

    def set_node_identity(
        self,
        node_id: int | None = None,
        node_ip: str | None = None,
        hostname: str | None = None,
        ray_node_id: str | None = None,
    ):
        """Update node identity used in emitted metric samples."""
        if node_id is not None:
            try:
                self.node_id = int(node_id)
            except (TypeError, ValueError):
                self.node_id = -1
        if node_ip:
            self.node_ip = str(node_ip)
        if hostname:
            self.hostname = str(hostname)
        if ray_node_id:
            self.ray_node_id = str(ray_node_id)

    @property
    def num_gpus(self) -> int:
        """Return the number of GPUs detected."""
        return self._num_gpus

    def _init_nvml(self):
        """Initialize NVML for GPU metrics collection."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._num_gpus = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) 
                for i in range(self._num_gpus)
            ]
            self._nvml_initialized = True
            _log.debug(f"NVML initialized with {self._num_gpus} GPUs")
        except Exception as e:
            _log.warning(f"NVML init failed: {e}")
            self._nvml_initialized = False
            self._num_gpus = 0
            self._gpu_handles = []

    def _collect_gpu_metrics(self, timestamp: float) -> list[GpuMetricSample]:
        """Collect metrics for all GPUs."""
        samples = []
        
        if not self._nvml_initialized:
            return samples

        try:
            import pynvml

            for gpu_idx, handle in enumerate(self._gpu_handles):
                try:
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_memory_used_gb",
                        value=mem_info.used / (1024**3),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_memory_free_gb",
                        value=mem_info.free / (1024**3),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                    if mem_info.total > 0:
                        samples.append(GpuMetricSample(
                            timestamp=timestamp,
                            gpu_index=gpu_idx,
                            metric_name="gpu_memory_used_percent",
                            value=(mem_info.used / mem_info.total) * 100.0,
                            node_id=self.node_id,
                            node_ip=self.node_ip,
                            hostname=self.hostname,
                            ray_node_id=self.ray_node_id,
                        ))
                except Exception:
                    pass

                try:
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_temperature_c",
                        value=float(temp),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                except Exception:
                    pass

                try:
                    # Power usage
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_power_w",
                        value=power,
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                except Exception:
                    pass

                try:
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_utilization_percent",
                        value=float(util.gpu),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_memory_bandwidth_utilization_percent",
                        value=float(util.memory),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                except Exception:
                    pass

                try:
                    # Clock speeds
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_SM
                    )
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_clock_sm_mhz",
                        value=float(sm_clock),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                except Exception:
                    pass

                try:
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM
                    )
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_clock_mem_mhz",
                        value=float(mem_clock),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                except Exception:
                    pass

                try:
                    # Fan speed (percentage)
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                    samples.append(GpuMetricSample(
                        timestamp=timestamp,
                        gpu_index=gpu_idx,
                        metric_name="gpu_fan_speed_percent",
                        value=float(fan_speed),
                        node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                    ))
                except Exception:
                    pass

        except Exception as e:
            _log.error(f"GPU metrics error: {e}")

        return samples

    def _collect_cpu_metrics(self, timestamp: float) -> list[CpuMetricSample]:
        """Collect CPU and system memory metrics."""
        samples = []

        # CPU utilization from /proc/stat
        try:
            cpu_percent = self._get_cpu_percent()
            if cpu_percent is not None:
                samples.append(CpuMetricSample(
                    timestamp=timestamp,
                    metric_name="cpu_utilization_percent",
                    value=cpu_percent,
                    node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                ))
        except Exception:
            pass

        # Per-core CPU utilization
        try:
            per_core = self._get_per_core_cpu_percent()
            for i, percent in enumerate(per_core):
                samples.append(CpuMetricSample(
                    timestamp=timestamp,
                    metric_name=f"cpu_core_{i}_utilization_percent",
                    value=percent,
                    node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                ))
        except Exception:
            pass

        # System memory from /proc/meminfo
        try:
            mem_info = self._get_memory_info()
            if mem_info:
                samples.append(CpuMetricSample(
                    timestamp=timestamp,
                    metric_name="system_memory_used_gb",
                    value=mem_info["used_gb"],
                    node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                ))
                samples.append(CpuMetricSample(
                    timestamp=timestamp,
                    metric_name="system_memory_available_gb",
                    value=mem_info["available_gb"],
                    node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                ))
                samples.append(CpuMetricSample(
                    timestamp=timestamp,
                    metric_name="system_memory_percent",
                    value=mem_info["percent"],
                    node_id=self.node_id,
                        node_ip=self.node_ip,
                        hostname=self.hostname,
                        ray_node_id=self.ray_node_id,
                ))
        except Exception:
            pass

        return samples

    # CPU percent tracking state
    _prev_cpu_times: tuple | None = None
    _prev_per_core_times: list | None = None

    def _get_cpu_percent(self) -> float | None:
        """Calculate CPU utilization percentage from /proc/stat."""
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()  # First line is aggregate CPU
            
            parts = line.split()
            if parts[0] != "cpu":
                return None

            # user, nice, system, idle, iowait, irq, softirq, steal
            times = tuple(int(x) for x in parts[1:9])
            
            if self._prev_cpu_times is None:
                self._prev_cpu_times = times
                return None

            # Calculate deltas
            deltas = tuple(t - p for t, p in zip(times, self._prev_cpu_times))
            self._prev_cpu_times = times

            total = sum(deltas)
            if total == 0:
                return 0.0

            # idle is index 3, iowait is index 4
            idle = deltas[3] + deltas[4]
            busy = total - idle
            return (busy / total) * 100.0

        except Exception:
            return None

    def _get_per_core_cpu_percent(self) -> list[float]:
        """Calculate per-core CPU utilization percentages."""
        try:
            with open("/proc/stat", "r") as f:
                lines = f.readlines()

            core_lines = [l for l in lines if l.startswith("cpu") and l[3].isdigit()]
            current_times = []
            
            for line in core_lines:
                parts = line.split()
                times = tuple(int(x) for x in parts[1:9])
                current_times.append(times)

            if self._prev_per_core_times is None:
                self._prev_per_core_times = current_times
                return []

            percents = []
            for curr, prev in zip(current_times, self._prev_per_core_times):
                deltas = tuple(t - p for t, p in zip(curr, prev))
                total = sum(deltas)
                if total == 0:
                    percents.append(0.0)
                else:
                    idle = deltas[3] + deltas[4]
                    busy = total - idle
                    percents.append((busy / total) * 100.0)

            self._prev_per_core_times = current_times
            return percents

        except Exception:
            return []

    def _get_memory_info(self) -> dict | None:
        """Get memory info from /proc/meminfo."""
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()

            mem = {}
            for line in lines:
                parts = line.split()
                key = parts[0].rstrip(":")
                value = int(parts[1])  # in kB
                mem[key] = value

            total = mem.get("MemTotal", 0) / (1024**2)  # kB to GB
            available = mem.get("MemAvailable", 0) / (1024**2)
            used = total - available
            percent = (used / total * 100) if total > 0 else 0

            return {
                "total_gb": total,
                "used_gb": used,
                "available_gb": available,
                "percent": percent,
            }

        except Exception:
            return None

    def _collect_all_metrics(self):
        """Collect all system metrics and add to buffers."""
        timestamp = time.time()
        gpu_samples = self._collect_gpu_metrics(timestamp)
        cpu_samples = self._collect_cpu_metrics(timestamp)
        
        with self._lock:
            self._gpu_metrics.extend(gpu_samples)
            self._cpu_metrics.extend(cpu_samples)

    def get_and_clear_metrics(self) -> tuple[list[GpuMetricSample], list[CpuMetricSample]]:
        """
        Get all collected metrics and clear the buffers.
        Called by EventLogger during upload cycle.
        
        Returns:
            Tuple of (gpu_metrics, cpu_metrics)
        """
        with self._lock:
            gpu_metrics = list(self._gpu_metrics)
            cpu_metrics = list(self._cpu_metrics)
            self._gpu_metrics = []
            self._cpu_metrics = []
        return gpu_metrics, cpu_metrics


    async def _collection_loop(self):
        """Background loop that collects metrics on the configured interval."""
        while not self._stop_event.is_set():
            try:
                self._collect_all_metrics()
            except Exception as e:
                _log.error(f"System metrics collection error: {e}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.collection_interval_seconds
                )
            except asyncio.TimeoutError:
                pass

    async def start(self):
        """Start the collection loop."""
        self._stop_event = asyncio.Event()
        self._collection_task = asyncio.create_task(self._collection_loop())
        _log.debug(
            f"System metrics collection started "
            f"(interval={self.collection_interval_seconds:.3f}s)"
        )

    async def stop(self):
        """Stop the collection loop."""
        if self._stop_event:
            self._stop_event.set()
        
        if self._collection_task:
            await self._collection_task
            self._collection_task = None
        
        _log.info("System metrics collection stopped")

    def finish(self):
        """Finalize and clean up resources."""
        # Shutdown NVML
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

        _log.info("SystemMetricsLogger finished")

"""
Executor and queue metrics logging for runtime concurrency monitoring.

This module collects utilization metrics for ThreadPoolExecutors,
ProcessPoolExecutors, and queues, then provides them to EventLogger for
unified upload to W&B.

Data is collected in the events folder:
- events/tail.zip: Contains thread_pools.parquet (along with other event parquet files)
- events/block_live.zip: Current 30-minute block
- events/block_*.zip: Finalized 30-minute blocks

Metrics are collected at a configurable interval (2 seconds by default).

Per-pool metrics (ThreadPoolExecutor):
- utilization_percent: active_threads / max_workers * 100
- active_threads: Number of threads currently alive in the pool
- max_workers: Maximum pool size (constant)
- queue_depth: Number of tasks waiting for a thread

Per-pool metrics (ProcessPoolExecutor):
- utilization_percent: active_processes / max_workers * 100
- active_processes: Number of processes currently alive
- max_workers: Maximum pool size (constant)
- queue_depth: Number of pending work items

Per-queue metrics:
- queue_depth: Number of items in the queue
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

from telescope.utils.tlog import get_logger

_log = get_logger("orchestrator")


@dataclass
class ThreadPoolMetricSample:
    """Single pool/queue metric sample."""
    timestamp: float
    pool_name: str
    metric_name: str
    value: float


class _TrackedThreadPool:
    """A ThreadPoolExecutor registered for metrics collection."""
    __slots__ = ("pool", "name", "max_workers")

    def __init__(self, pool: ThreadPoolExecutor, name: str, max_workers: int):
        self.pool = pool
        self.name = name
        self.max_workers = max_workers


class _TrackedProcessPool:
    """A ProcessPoolExecutor registered for metrics collection."""
    __slots__ = ("pool", "name", "max_workers")

    def __init__(self, pool: ProcessPoolExecutor, name: str, max_workers: int):
        self.pool = pool
        self.name = name
        self.max_workers = max_workers


class _QueueTracker:
    """A queue registered for depth monitoring."""
    __slots__ = ("queue_obj", "name")

    def __init__(self, queue_obj, name: str):
        self.queue_obj = queue_obj
        self.name = name


class ThreadPoolMetricsLogger:
    """
    Thread-safe metrics logger that collects executor/queue utilization
    at a configurable interval and provides data to EventLogger for unified upload.

    Usage:
        logger = ThreadPoolMetricsLogger()
        logger.register_thread_pool(http_pool, "http", max_workers=2048)
        logger.register_process_pool(batch_pool, "batch_executor", max_workers=1)
        logger.register_queue(sandbox_queue, "sandbox_ready")

        await logger.start()

        # Get metrics for upload (called by EventLogger)
        metrics = logger.get_and_clear_metrics()

        await logger.stop()
    """

    DEFAULT_COLLECTION_INTERVAL_SECONDS = 2.0

    def __init__(self, collection_interval_seconds: float = DEFAULT_COLLECTION_INTERVAL_SECONDS):
        self._lock = threading.Lock()
        self._tracked: list[_TrackedThreadPool | _TrackedProcessPool | _QueueTracker] = []
        self._metrics: list[ThreadPoolMetricSample] = []
        self.collection_interval_seconds = max(0.5, float(collection_interval_seconds))

        # Background task
        self._stop_event: asyncio.Event | None = None
        self._collection_task: asyncio.Task | None = None

    # Keep _pools as alias for backward compat with orchestrator registration code
    @property
    def _pools(self):
        return self._tracked

    def register_pool(
        self,
        pool: ThreadPoolExecutor,
        name: str,
        max_workers: int,
    ):
        """Register a ThreadPoolExecutor for metrics collection."""
        self.register_thread_pool(pool, name, max_workers)

    def register_thread_pool(
        self,
        pool: ThreadPoolExecutor,
        name: str,
        max_workers: int,
    ):
        """Register a ThreadPoolExecutor for metrics collection."""
        self._tracked.append(_TrackedThreadPool(pool=pool, name=name, max_workers=max_workers))
        _log.debug(f"Registered thread pool '{name}' (max_workers={max_workers})")

    def register_process_pool(
        self,
        pool: ProcessPoolExecutor,
        name: str,
        max_workers: int,
    ):
        """Register a ProcessPoolExecutor for metrics collection."""
        self._tracked.append(_TrackedProcessPool(pool=pool, name=name, max_workers=max_workers))
        _log.debug(f"Registered process pool '{name}' (max_workers={max_workers})")

    def register_queue(
        self,
        queue_obj,
        name: str,
    ):
        """Register a queue for depth monitoring.

        The queue must have a ``qsize()`` method (e.g. ``queue.Queue``).
        """
        self._tracked.append(_QueueTracker(queue_obj=queue_obj, name=name))
        _log.debug(f"Registered queue '{name}' for depth monitoring")

    def _collect_all_metrics(self):
        """Snapshot all registered executors and queues."""
        timestamp = time.time()
        samples: list[ThreadPoolMetricSample] = []

        def _sample(pool_name: str, metric_name: str, value: float):
            samples.append(ThreadPoolMetricSample(
                timestamp=timestamp,
                pool_name=pool_name,
                metric_name=metric_name,
                value=value,
            ))

        for tracked in self._tracked:
            if isinstance(tracked, _QueueTracker):
                try:
                    depth = tracked.queue_obj.qsize()
                except Exception:
                    continue
                _sample(tracked.name, "queue_depth", float(depth))
                continue

            if isinstance(tracked, _TrackedProcessPool):
                pool = tracked.pool
                try:
                    # ProcessPoolExecutor._processes is a dict {pid: process}
                    active = len(getattr(pool, "_processes", {}))
                except Exception:
                    active = 0
                try:
                    queue_depth = len(getattr(pool, "_pending_work_items", {}))
                except Exception:
                    queue_depth = 0

                _sample(tracked.name, "active_processes", float(active))
                _sample(tracked.name, "max_workers", float(tracked.max_workers))
                _sample(tracked.name, "queue_depth", float(queue_depth))
                if tracked.max_workers > 0:
                    _sample(tracked.name, "utilization_percent",
                            active / tracked.max_workers * 100.0)
                continue

            # ThreadPoolExecutor
            pool = tracked.pool
            try:
                queue_depth = pool._work_queue.qsize()
            except Exception:
                queue_depth = 0

            try:
                active = len(pool._threads)
            except Exception:
                active = 0

            _sample(tracked.name, "active_threads", float(active))
            _sample(tracked.name, "max_workers", float(tracked.max_workers))
            _sample(tracked.name, "queue_depth", float(queue_depth))
            if tracked.max_workers > 0:
                _sample(tracked.name, "utilization_percent",
                        active / tracked.max_workers * 100.0)

        with self._lock:
            self._metrics.extend(samples)

    def get_and_clear_metrics(self) -> list[ThreadPoolMetricSample]:
        """
        Get all collected metrics and clear the buffer.
        Called by EventLogger during upload cycle.
        """
        with self._lock:
            metrics = list(self._metrics)
            self._metrics = []
        return metrics

    async def _collection_loop(self):
        """Background loop that collects metrics on the configured interval."""
        while not self._stop_event.is_set():
            try:
                self._collect_all_metrics()
            except Exception as e:
                _log.error(f"Thread pool metrics collection error: {e}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.collection_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def start(self):
        """Start the collection loop."""
        self._stop_event = asyncio.Event()
        self._collection_task = asyncio.create_task(self._collection_loop())
        names = [t.name for t in self._tracked]
        _log.debug(
            f"Thread pool metrics collection started "
            f"(interval={self.collection_interval_seconds:.1f}s, tracked={names})"
        )

    async def stop(self):
        """Stop the collection loop."""
        if self._stop_event:
            self._stop_event.set()
        if self._collection_task:
            await self._collection_task
            self._collection_task = None
        _log.info("Thread pool metrics collection stopped")

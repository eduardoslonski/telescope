"""
Thread pool metrics logging for runtime concurrency monitoring.

This module collects thread pool utilization metrics and provides them
to EventLogger for unified upload to W&B. Data is collected in the events folder:
- events/tail.zip: Contains thread_pools.parquet (along with other event parquet files)
- events/block_live.zip: Current 30-minute block
- events/block_*.zip: Finalized 30-minute blocks

Metrics are collected at a configurable interval (2 seconds by default).

Per-pool metrics:
- active_threads: Number of threads currently executing tasks
- max_workers: Maximum pool size
- queue_depth: Number of tasks waiting for a thread
- total_threads: Number of alive threads in the pool
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from telescope.utils.tlog import get_logger

_log = get_logger("orchestrator")


@dataclass
class ThreadPoolMetricSample:
    """Single thread pool metric sample."""
    timestamp: float
    pool_name: str
    metric_name: str
    value: float


class _TrackedPool:
    """A thread pool registered for metrics collection."""
    __slots__ = ("pool", "name", "max_workers")

    def __init__(self, pool: ThreadPoolExecutor, name: str, max_workers: int):
        self.pool = pool
        self.name = name
        self.max_workers = max_workers


class ThreadPoolMetricsLogger:
    """
    Thread-safe thread pool metrics logger that collects pool utilization
    at a configurable interval and provides data to EventLogger for unified upload.

    Usage:
        logger = ThreadPoolMetricsLogger()
        logger.register_pool(http_pool, "http", max_workers=2048)
        logger.register_pool(tar_pool, "tar_builder", max_workers=128)

        await logger.start()

        # Get metrics for upload (called by EventLogger)
        metrics = logger.get_and_clear_metrics()

        await logger.stop()
    """

    DEFAULT_COLLECTION_INTERVAL_SECONDS = 2.0

    def __init__(self, collection_interval_seconds: float = DEFAULT_COLLECTION_INTERVAL_SECONDS):
        self._lock = threading.Lock()
        self._pools: list[_TrackedPool] = []
        self._metrics: list[ThreadPoolMetricSample] = []
        self.collection_interval_seconds = max(0.5, float(collection_interval_seconds))

        # Background task
        self._stop_event: asyncio.Event | None = None
        self._collection_task: asyncio.Task | None = None

    def register_pool(
        self,
        pool: ThreadPoolExecutor,
        name: str,
        max_workers: int,
    ):
        """Register a thread pool for metrics collection."""
        self._pools.append(_TrackedPool(pool=pool, name=name, max_workers=max_workers))
        _log.debug(f"Registered thread pool '{name}' (max_workers={max_workers})")

    def register_queue(
        self,
        queue_obj,
        name: str,
    ):
        """Register a queue for depth monitoring.

        The queue must have a ``qsize()`` method (e.g. ``queue.Queue``).
        Internally stored as a pseudo-pool so the collection loop handles
        it uniformly via duck-typing.
        """
        self._pools.append(_QueueTracker(queue_obj=queue_obj, name=name))
        _log.debug(f"Registered queue '{name}' for depth monitoring")

    def _collect_all_metrics(self):
        """Snapshot all registered pools and queues."""
        timestamp = time.time()
        samples: list[ThreadPoolMetricSample] = []

        for tracked in self._pools:
            if isinstance(tracked, _QueueTracker):
                try:
                    depth = tracked.queue_obj.qsize()
                except Exception:
                    continue
                samples.append(ThreadPoolMetricSample(
                    timestamp=timestamp,
                    pool_name=tracked.name,
                    metric_name="queue_depth",
                    value=float(depth),
                ))
                continue

            pool = tracked.pool
            # ThreadPoolExecutor internals — stable across CPython 3.8+
            try:
                queue_depth = pool._work_queue.qsize()
            except Exception:
                queue_depth = 0

            try:
                total_threads = len(pool._threads)
            except Exception:
                total_threads = 0

            # Count active threads by name prefix (threads that are alive and
            # actually running, not idle waiting on the work queue).  We use
            # the thread_name_prefix set on each pool to filter.
            prefix = getattr(pool, "_thread_name_prefix", None) or ""
            if prefix:
                active = sum(
                    1 for t in threading.enumerate()
                    if t.name.startswith(prefix) and t.is_alive()
                )
            else:
                active = total_threads

            samples.append(ThreadPoolMetricSample(
                timestamp=timestamp,
                pool_name=tracked.name,
                metric_name="active_threads",
                value=float(active),
            ))
            samples.append(ThreadPoolMetricSample(
                timestamp=timestamp,
                pool_name=tracked.name,
                metric_name="max_workers",
                value=float(tracked.max_workers),
            ))
            samples.append(ThreadPoolMetricSample(
                timestamp=timestamp,
                pool_name=tracked.name,
                metric_name="queue_depth",
                value=float(queue_depth),
            ))
            samples.append(ThreadPoolMetricSample(
                timestamp=timestamp,
                pool_name=tracked.name,
                metric_name="total_threads",
                value=float(total_threads),
            ))

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
        pool_names = [p.name for p in self._pools]
        _log.debug(
            f"Thread pool metrics collection started "
            f"(interval={self.collection_interval_seconds:.1f}s, pools={pool_names})"
        )

    async def stop(self):
        """Stop the collection loop."""
        if self._stop_event:
            self._stop_event.set()
        if self._collection_task:
            await self._collection_task
            self._collection_task = None
        _log.info("Thread pool metrics collection stopped")


class _QueueTracker:
    """Lightweight wrapper so queues can sit in the same ``_pools`` list."""
    __slots__ = ("queue_obj", "name")

    def __init__(self, queue_obj, name: str):
        self.queue_obj = queue_obj
        self.name = name

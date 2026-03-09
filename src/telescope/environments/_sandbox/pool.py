"""
Generic sandbox pool — provider-agnostic producer-consumer pattern.

Replaces both ``i3_code.sandbox_pool.SandboxPool`` and
``i3_code_modal.sandbox_pool.ModalSandboxPool`` with a single
implementation that delegates sandbox lifecycle to a
:class:`SandboxProvider`.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SandboxConfig, SandboxHandle, SandboxProvider


class GenericSandboxPool:
    """
    Provider-agnostic producer-consumer pool for pre-created sandboxes.

    Architecture:
    - Producer thread: Separate thread + event loop, creates sandboxes
      via ``await provider.create(config)``
    - Main event loop: Orchestrates rollouts, acquires pre-warmed sandboxes
    - Sandboxes are cleaned and returned to pool for reuse

    Pool sizing: Size based on concurrent TEST EXECUTIONS, not total
    rollouts.  Sandboxes are only acquired during test execution (5-15s),
    so you can support many more rollouts than pool_size.

    Cleanup: Disabled for performance.  Bundles use unique names (UUIDs)
    so no conflicts on reuse.
    """

    def __init__(
        self,
        provider: SandboxProvider,
        config: SandboxConfig,
        pool_size: int = 10,
        max_concurrent_creates: int = 100,
    ):
        self.provider = provider
        self.config = config
        self.pool_size = pool_size
        self.max_concurrent_creates = max_concurrent_creates
        self.timeout_seconds = config.timeout_seconds

        # Thread-safe queue for ready sandbox handles
        self.ready_queue: queue.Queue[SandboxHandle] = queue.Queue(maxsize=pool_size)

        # Track all sandboxes (thread-safe via lock)
        self._lock = threading.Lock()
        self.all_sandboxes: dict[str, SandboxHandle] = {}  # id -> handle
        self.in_use_sandboxes: set[str] = set()  # ids
        self.sandbox_creation_times: dict[str, float] = {}  # id -> timestamp
        self.pending_creates: int = 0

        # Rate limit "waiting for sandbox" log spam
        self._last_waiting_log = 0.0

        # Producer thread
        self.producer_thread: threading.Thread | None = None
        self.producer_loop: asyncio.AbstractEventLoop | None = None
        self.shutdown_event = threading.Event()
        self._started = False
        self._start_lock = threading.Lock()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def start(self):
        """Start the producer thread (idempotent)."""
        with self._start_lock:
            if self._started:
                return

            self.logger.info(
                f"Starting sandbox pool producer thread (pool_size={self.pool_size})"
            )
            self.producer_thread = threading.Thread(
                target=self._run_producer_thread,
                daemon=True,
                name="SandboxPoolProducer",
            )
            self.producer_thread.start()
            self._started = True

    def _run_producer_thread(self):
        """Entry point for producer thread — creates its own event loop."""
        try:
            self.producer_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.producer_loop)
            self.logger.debug("Producer thread started with dedicated event loop")
            self.producer_loop.run_until_complete(self._producer_loop())
        except Exception as e:
            self.logger.error(f"Producer thread crashed: {e!r}")
            traceback.print_exc()
        finally:
            if self.producer_loop:
                self.producer_loop.close()
            self.logger.debug("Producer thread exiting")

    async def _producer_loop(self):
        """Continuously create sandboxes to maintain pool size."""
        last_pool_status_log = 0.0
        pool_status_log_interval = 5.0
        semaphore = asyncio.Semaphore(self.max_concurrent_creates)

        while not self.shutdown_event.is_set():
            try:
                with self._lock:
                    total = len(self.all_sandboxes)
                    in_use = len(self.in_use_sandboxes)
                    pending = self.pending_creates
                    effective_total = total + pending

                ready_count = self.ready_queue.qsize()
                needed = self.pool_size - effective_total

                # Log pool status periodically
                current_time = time.time()
                if current_time - last_pool_status_log >= pool_status_log_interval:
                    status_parts = [
                        f"{ready_count} ready",
                        f"{in_use} in-use",
                        f"{total}/{self.pool_size} total",
                    ]
                    if pending > 0:
                        status_parts.append(f"{pending} preparing")
                    if needed > 0:
                        status_parts.append(f"need {needed} more")
                    self.logger.debug(f"Pool: {', '.join(status_parts)}")
                    last_pool_status_log = current_time

                if needed > 0:
                    batch_size = min(needed, self.max_concurrent_creates)

                    with self._lock:
                        self.pending_creates += batch_size

                    self.logger.debug(
                        f"Producer: Creating batch of {batch_size} sandboxes..."
                    )

                    batch_start = time.perf_counter()

                    async def _create_one():
                        if self.shutdown_event.is_set():
                            return None
                        async with semaphore:
                            return await self.provider.create(self.config)

                    tasks = [_create_one() for _ in range(batch_size)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    successful = 0
                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.error(
                                f"Producer: error creating sandbox: {result!r}"
                            )
                            with self._lock:
                                self.pending_creates -= 1
                        elif result is not None:
                            handle = result
                            with self._lock:
                                self.all_sandboxes[handle.id] = handle
                                self.sandbox_creation_times[handle.id] = time.time()
                                self.pending_creates -= 1
                            self.ready_queue.put(handle)
                            successful += 1
                        else:
                            with self._lock:
                                self.pending_creates -= 1

                    batch_time = time.perf_counter() - batch_start
                    if successful > 0:
                        self.logger.debug(
                            f"Batch complete: {successful}/{batch_size} sandboxes "
                            f"ready in {batch_time:.2f}s "
                            f"({batch_time / successful:.2f}s avg)"
                        )

                    # Verify invariants
                    with self._lock:
                        if self.pending_creates < 0:
                            self.logger.error(
                                f"CRITICAL: pending_creates is negative "
                                f"({self.pending_creates}), resetting to 0"
                            )
                            self.pending_creates = 0
                        if len(self.all_sandboxes) > self.pool_size:
                            self.logger.error(
                                f"CRITICAL: all_sandboxes exceeds pool_size "
                                f"({len(self.all_sandboxes)} > {self.pool_size})"
                            )
                else:
                    await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Producer: error in loop: {e!r}")
                traceback.print_exc()
                await asyncio.sleep(5.0)

        self.logger.debug("Producer loop exiting")

    async def acquire(self, timeout: float | None = None) -> SandboxHandle:
        """Acquire a sandbox from the pool (non-blocking polling)."""
        start_time = time.monotonic()

        while True:
            try:
                handle = self.ready_queue.get_nowait()
            except queue.Empty:
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        raise TimeoutError(
                            f"Failed to acquire sandbox within {timeout}s timeout"
                        )

                current_time = time.time()
                if current_time - self._last_waiting_log >= 5.0:
                    with self._lock:
                        in_use = len(self.in_use_sandboxes)
                        pending = self.pending_creates
                    total = len(self.all_sandboxes)
                    if total == 0:
                        self.logger.info(
                            f"Waiting for initial sandbox pool fill "
                            f"({pending} creating) — this may take a "
                            f"few minutes on first run (image pull)"
                        )
                    else:
                        self.logger.warning(
                            f"Pool exhausted! 0 ready, {in_use} in-use, "
                            f"{total} total, {pending} preparing — "
                            f"rollouts are waiting"
                        )
                    self._last_waiting_log = current_time

                await asyncio.sleep(0.05)
                continue

            # Check age
            with self._lock:
                creation_time = self.sandbox_creation_times.get(handle.id)

            if creation_time:
                age_seconds = time.time() - creation_time
                remaining = self.timeout_seconds - age_seconds

                if remaining < self.timeout_seconds * 0.1:
                    self.logger.warning(
                        f"Sandbox {handle.id} too old "
                        f"(age: {age_seconds / 60:.1f}m), removing"
                    )
                    await self.remove(handle)
                    continue

            with self._lock:
                self.in_use_sandboxes.add(handle.id)

            return handle

    async def release(self, handle: SandboxHandle):
        """Release a sandbox back to the pool for reuse."""
        with self._lock:
            self.in_use_sandboxes.discard(handle.id)

            if handle.id not in self.all_sandboxes:
                self.logger.error(
                    f"Attempted to release unknown sandbox {handle.id}"
                )
                return

            creation_time = self.sandbox_creation_times.get(handle.id)

        if creation_time:
            age_seconds = time.time() - creation_time
            remaining = self.timeout_seconds - age_seconds

            if remaining < self.timeout_seconds * 0.2:
                self.logger.info(
                    f"Sandbox {handle.id} nearing timeout, removing"
                )
                await self.remove(handle)
                return

        try:
            self.ready_queue.put_nowait(handle)
        except queue.Full:
            self.logger.error(
                f"CRITICAL: Pool queue full, destroying sandbox {handle.id}"
            )
            with self._lock:
                self.all_sandboxes.pop(handle.id, None)
                self.sandbox_creation_times.pop(handle.id, None)
            try:
                await self.provider.destroy(handle)
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f"Failed to return {handle.id} to pool: {e!r}")
            with self._lock:
                self.all_sandboxes.pop(handle.id, None)
                self.sandbox_creation_times.pop(handle.id, None)
            try:
                await self.provider.destroy(handle)
            except Exception:
                pass

    async def remove(self, handle: SandboxHandle):
        """Remove a dead/failed sandbox from the pool and destroy it."""
        with self._lock:
            self.in_use_sandboxes.discard(handle.id)
            self.all_sandboxes.pop(handle.id, None)
            creation_time = self.sandbox_creation_times.pop(handle.id, None)

        age_minutes = (time.time() - creation_time) / 60 if creation_time else None
        age_str = f" (age: {age_minutes:.1f}m)" if age_minutes is not None else ""
        self.logger.warning(
            f"Removed sandbox {handle.id}{age_str}, "
            f"producer will create replacement"
        )

        try:
            await self.provider.destroy(handle)
        except Exception as e:
            self.logger.warning(
                f"Failed to destroy sandbox {handle.id}: {e!r}"
            )

    async def shutdown(self):
        """Shutdown the producer thread and destroy all sandboxes."""
        self.logger.info("Shutting down sandbox pool...")

        self.shutdown_event.set()

        if self.producer_thread is not None and self.producer_thread.is_alive():
            self.logger.debug("Waiting for producer thread to exit...")
            self.producer_thread.join(timeout=10.0)
            if self.producer_thread.is_alive():
                self.logger.warning("Producer thread did not exit cleanly")

        with self._lock:
            all_handles = list(self.all_sandboxes.values())

        if all_handles:
            self.logger.info(
                f"Destroying {len(all_handles)} sandboxes..."
            )
            await self.provider.bulk_destroy(all_handles)

        with self._lock:
            self.all_sandboxes.clear()
            self.in_use_sandboxes.clear()
            self.sandbox_creation_times.clear()

        self.logger.info("Sandbox pool shutdown complete")

    def shutdown_sync(self):
        """Synchronous fallback shutdown (for atexit handler)."""
        self.shutdown_event.set()

        with self._lock:
            all_handles = list(self.all_sandboxes.values())

        if not all_handles:
            return

        self.logger.debug(
            f"Cleaning up {len(all_handles)} sandboxes via sync fallback"
        )
        self.provider.bulk_destroy_sync(all_handles)

        with self._lock:
            self.all_sandboxes.clear()
            self.in_use_sandboxes.clear()
            self.sandbox_creation_times.clear()

        self.logger.debug("Sync cleanup complete")

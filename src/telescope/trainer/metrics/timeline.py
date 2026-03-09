"""GPU timeline tracking for Gantt chart visualization."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Generator


def _parse_event_hierarchy(event_type: str) -> tuple[str | None, int]:
    """
    Parse event_type into parent and depth.
    
    Returns:
        (parent, depth) where parent is None for root events.
        
    Examples:
        "forward" -> (None, 0)
        "loss" -> (None, 0)
        "loss/log_softmax" -> ("loss", 1)
        "loss/policy/core" -> ("loss/policy", 2)
    """
    if "/" not in event_type:
        return None, 0
    
    parts = event_type.rsplit("/", 1)
    parent = parts[0]
    depth = event_type.count("/")
    return parent, depth


@dataclass
class GPUEvent:
    """Single GPU event for timeline visualization."""
    event_type: str  # "forward", "loss", "loss/log_softmax", etc.
    start_time: float  # Absolute Unix timestamp
    end_time: float  # Absolute Unix timestamp
    rank: int = 0
    parent: str | None = None  # Parent event path, None for root events
    depth: int = 0  # Nesting level: 0 for root, 1 for first-level children, etc.
    microbatch: int = -1  # Micro batch index (-1 if not a per-microbatch event)
    minibatch: int = -1  # Mini batch index (-1 if not applicable or only 1 minibatch)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class GPUTimelineLogger:
    """
    Tracks GPU events during a training step for Gantt chart visualization,
    using CUDA events for accurate GPU timing (low perturbation).

    Notes:
    - CUDA events measure time on the current CUDA stream.
    - We synchronize once per step in finalize_step() to materialize timings.
    """

    def __init__(self, rank: int = 0, device: torch.device | None = None):
        self.rank = rank
        self.device = device

        self.step_start_wall_time: float = 0.0
        self._step_perf_start: float = 0.0
        self.events: list[GPUEvent] = []
        self._context_stack: list[str] = []

        # CUDA timing state
        self._cuda_enabled: bool = torch.cuda.is_available()
        self._step_cuda_start: torch.cuda.Event | None = None
        self._pending_cuda: list[tuple[str, torch.cuda.Event, torch.cuda.Event, int, int]] = []

    def start_step(self):
        """Call at the beginning of each step to reset and set baseline time."""
        self.step_start_wall_time = time.time()
        self._step_perf_start = time.perf_counter()
        self.events = []
        self._context_stack = []
        self._pending_cuda = []

        if self._cuda_enabled:
            # Record a reference event to convert later elapsed_time -> wall time.
            self._step_cuda_start = torch.cuda.Event(enable_timing=True)
            self._step_cuda_start.record()  # current stream

    @contextmanager
    def track(
        self,
        name: str,
        cpu: bool = False,
        microbatch: int = -1,
        minibatch: int = -1,
        nest: bool = True,
    ) -> Generator[None, None, None]:
        """
        Context manager for tracking events with automatic timing.
        Supports nesting - child events are auto-prefixed with parent name.

        Args:
            name: Event name (will be prefixed with parent if nested)
            cpu: If True, use CPU wall-clock timing instead of CUDA events.
                 Use this for CPU-only operations like metrics collection, file I/O, etc.
            microbatch: Micro batch index (-1 if not a per-microbatch event)
            minibatch: Mini batch index (-1 if not applicable or only 1 minibatch)
            nest: If True (default), nest under the current context and prefix child
                  event names. If False, record this event as top-level without
                  changing child event naming.
        """
        full_name = (
            f"{self._context_stack[-1]}/{name}"
            if nest and self._context_stack
            else name
        )
        if nest:
            self._context_stack.append(full_name)

        if self._cuda_enabled and not cpu:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()  # current stream
            try:
                yield
            finally:
                end_ev.record()  # current stream
                self._pending_cuda.append((full_name, start_ev, end_ev, microbatch, minibatch))
                if nest:
                    self._context_stack.pop()
        else:
            # CPU timing for cpu=True or when CUDA not available
            start = time.perf_counter()
            try:
                yield
            finally:
                end = time.perf_counter()
                if nest:
                    self._context_stack.pop()
                self._log_cpu_event(full_name, start, end, microbatch, minibatch)

    def finalize_step(self):
        """
        Convert recorded CUDA events into GPUEvent objects.
        Must be called once per step after all tracked regions finished.
        """
        if not self._cuda_enabled or self._step_cuda_start is None:
            return

        # Materialize all GPU work so elapsed_time is valid
        torch.cuda.synchronize()

        for event_type, start_ev, end_ev, microbatch, minibatch in self._pending_cuda:
            # ms from step start
            start_ms = self._step_cuda_start.elapsed_time(start_ev)
            dur_ms = start_ev.elapsed_time(end_ev)

            start_wall = self.step_start_wall_time + (start_ms / 1000.0)
            end_wall = start_wall + (dur_ms / 1000.0)

            parent, depth = _parse_event_hierarchy(event_type)
            self.events.append(GPUEvent(
                event_type=event_type,
                start_time=start_wall,
                end_time=end_wall,
                rank=self.rank,
                parent=parent,
                depth=depth,
                microbatch=microbatch,
                minibatch=minibatch,
            ))

        # Clear pending list to avoid double finalization
        self._pending_cuda = []

    def _log_cpu_event(self, event_type: str, start_time: float, end_time: float, microbatch: int = -1, minibatch: int = -1):
        # CPU fallback uses perf_counter anchored to wall time.
        # start_wall ~= step_start_wall + (start_perf - step_start_perf)
        start_wall = self.step_start_wall_time + (start_time - self._step_perf_start)
        end_wall = start_wall + (end_time - start_time)
        parent, depth = _parse_event_hierarchy(event_type)
        self.events.append(GPUEvent(
            event_type=event_type,
            start_time=start_wall,
            end_time=end_wall,
            rank=self.rank,
            parent=parent,
            depth=depth,
            microbatch=microbatch,
            minibatch=minibatch,
        ))

    def get_serializable_events(self) -> list[dict]:
        return [
            {
                "event_type": e.event_type,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "rank": e.rank,
                "parent": e.parent,
                "depth": e.depth,
                "microbatch": e.microbatch,
                "minibatch": e.minibatch,
            }
            for e in self.events
        ]



class _NullTracker:
    """No-op tracker for when timeline logging is disabled."""
    
    events: list[GPUEvent] = []
    
    def start_step(self):
        pass
    
    @contextmanager
    def track(
        self,
        name: str,
        cpu: bool = False,
        microbatch: int = -1,
        minibatch: int = -1,
        nest: bool = True,
    ) -> Generator[None, None, None]:
        yield


# Singleton instance for null tracking
_null_tracker = _NullTracker()


def create_timeline_tracker(
    timeline_logger: GPUTimelineLogger | None,
) -> GPUTimelineLogger | _NullTracker:
    """
    Create a timeline tracker that works even when logger is None.
    
    Returns the logger if provided, otherwise returns a no-op tracker
    that has the same interface but does nothing.
    """
    return timeline_logger if timeline_logger else _null_tracker

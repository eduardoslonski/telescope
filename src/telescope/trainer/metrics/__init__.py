"""
Trainer metrics module.

Provides tracking of:
- GPU timeline for Gantt chart visualization
"""
from telescope.trainer.metrics.timeline import (
    GPUTimelineLogger,
    create_timeline_tracker,
)

__all__ = [
    "GPUTimelineLogger",
    "create_timeline_tracker",
]


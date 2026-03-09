"""Orchestrator package.

Keep package import side-effect free to avoid circular imports in Ray workers.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Orchestrator", "main"]


def __getattr__(name: str) -> Any:
    """Lazily expose orchestrator entrypoints."""
    if name in {"Orchestrator", "main"}:
        from telescope.orchestrator.orchestrator import Orchestrator, main

        if name == "Orchestrator":
            return Orchestrator
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


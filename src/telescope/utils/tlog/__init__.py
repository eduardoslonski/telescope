"""
Telescope logging system.

Provides structured logging for orchestrator, trainer, and inference components.
Each component gets its own log directory with standard and detailed log files.
"""
from telescope.utils.tlog.logger import (
    get_logger,
    is_debug_mode,
    setup_logging,
    LogLevel,
    TelescopeLogger,
)

__all__ = [
    "get_logger",
    "is_debug_mode",
    "setup_logging",
    "LogLevel",
    "TelescopeLogger",
]



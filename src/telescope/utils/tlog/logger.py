"""
Professional logging system for Telescope.

Features:
- Separate log directories for orchestrator, trainer, and inference
- Standard log file (clean, essential info) + detailed log file (verbose debugging)
- Structured output with timestamps, levels, and context
- Thread-safe operation
- Support for step/rank context in distributed training
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from enum import IntEnum
from pathlib import Path

# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright variants
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class LogLevel(IntEnum):
    """Log levels matching Python's logging module."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Log level colors
LEVEL_COLORS = {
    logging.DEBUG: Colors.BRIGHT_BLACK,
    logging.INFO: Colors.BRIGHT_CYAN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD,
}

# Component colors for terminal
COMPONENT_COLORS = {
    "orchestrator": Colors.BRIGHT_MAGENTA,
    "trainer": Colors.BRIGHT_GREEN,
    "inference": Colors.BRIGHT_BLUE,
}


class TelescopeFormatter(logging.Formatter):
    """
    Custom formatter for Telescope logs.

    Console format:  HH:MM:SS [ INFO ] message
    File format:     [HH:MM:SS.mmm] [LEVEL] [component] [step=N] message
    """

    def __init__(self, use_colors: bool = True, detailed: bool = False, console: bool = False):
        super().__init__()
        self.use_colors = use_colors
        self.detailed = detailed
        self.console = console

    def format(self, record: logging.LogRecord) -> str:
        # Build the message
        message = record.getMessage()

        # Add exception info if present (for detailed logs)
        if self.detailed and record.exc_info:
            exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{exc_text}"

        # --- Console format: clean, no component, no alignment padding ---
        if self.console:
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

            if self.use_colors:
                level_color = LEVEL_COLORS.get(record.levelno, "")
                return (
                    f"{Colors.DIM}{timestamp}{Colors.RESET} "
                    f"{level_color}[{record.levelname}]{Colors.RESET} "
                    f"{message}"
                )
            else:
                return f"{timestamp} [{record.levelname}] {message}"

        # --- File format: verbose with component and context ---
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        level_name = record.levelname.ljust(8)

        parts = record.name.split(".")
        component = parts[1] if len(parts) > 1 else parts[0]
        component = component.ljust(12)

        # Optional context (step, rank)
        context = ""
        step = getattr(record, "step", None)
        rank = getattr(record, "rank", None)
        if step is not None or rank is not None:
            context_parts = []
            if step is not None:
                context_parts.append(f"step={step}")
            if rank is not None:
                context_parts.append(f"rank={rank}")
            context = f" [{', '.join(context_parts)}]"

        return f"[{timestamp}] [{level_name}] [{component}]{context} {message}"


class DetailedFormatter(TelescopeFormatter):
    """Extended formatter for detailed logs with extra debugging info."""
    
    def __init__(self, use_colors: bool = False):
        super().__init__(use_colors=use_colors, detailed=True)
    
    def format(self, record: logging.LogRecord) -> str:
        # Get base format
        base = super().format(record)
        
        # Add file/line info for detailed logs
        location = f"{record.filename}:{record.lineno}"
        
        # Add thread info if relevant
        thread_info = ""
        if record.thread:
            thread_info = f" [thread={record.threadName}]"
        
        # Add function name
        func_info = f" in {record.funcName}()" if record.funcName else ""
        
        return f"{base}  [{location}{func_info}{thread_info}]"


class TelescopeLogger:
    """
    Logger wrapper with convenience methods for Telescope components.
    
    Provides:
    - Standard logging methods (debug, info, warning, error, critical)
    - Context-aware logging with step/rank
    - Banner/section logging for visual organization
    """
    
    def __init__(self, logger: logging.Logger, component: str):
        self._logger = logger
        self._component = component
        self._default_step: int | None = None
        self._default_rank: int | None = None
    
    def _log(
        self,
        level: int,
        message: str,
        step: int | None = None,
        rank: int | None = None,
        **kwargs,
    ):
        """Internal logging method with context support."""
        # Use provided step/rank or fall back to defaults
        actual_step = step if step is not None else self._default_step
        actual_rank = rank if rank is not None else self._default_rank
        
        extra = {
            "step": actual_step,
            "rank": actual_rank,
        }
        self._logger.log(level, message, extra=extra, **kwargs)
    
    def debug(self, message: str, step: int = None, rank: int = None, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, step=step, rank=rank, **kwargs)
    
    def info(self, message: str, step: int = None, rank: int = None, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, step=step, rank=rank, **kwargs)
    
    def warning(self, message: str, step: int = None, rank: int = None, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, step=step, rank=rank, **kwargs)
    
    def error(self, message: str, step: int = None, rank: int = None, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, step=step, rank=rank, **kwargs)
    
    def critical(self, message: str, step: int = None, rank: int = None, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, step=step, rank=rank, **kwargs)
    
    def exception(self, message: str, step: int = None, rank: int = None, **kwargs):
        """Log exception with traceback."""
        self._log(logging.ERROR, message, step=step, rank=rank, exc_info=True, **kwargs)
    
    def banner(self, title: str, char: str = "═", width: int = 70):
        """
        Log a banner-style section header.
        
        Example:
            ══════════════════════════════════════════════════════════════════════
            Training Started
            ══════════════════════════════════════════════════════════════════════
        """
        line = char * width
        self.info(line)
        self.info(title)
        self.info(line)
    
    def section(self, title: str):
        """
        Log a section header (lighter than banner).
        
        Example:
            ── Initializing Model ──
        """
        self.info(f"── {title} ──")
    
    def metric(self, name: str, value: float, step: int = None, rank: int = None, fmt: str = ".4f"):
        """Log a metric value."""
        self.info(f"{name}: {value:{fmt}}", step=step, rank=rank)
    
    def metrics(self, metrics: dict, step: int = None, rank: int = None, fmt: str = ".4f"):
        """Log multiple metrics on one line."""
        parts = [f"{k}: {v:{fmt}}" for k, v in metrics.items()]
        self.info(" | ".join(parts), step=step, rank=rank)
    
    def timing(self, operation: str, duration: float, step: int = None, rank: int = None):
        """Log an operation timing."""
        self.info(f"{operation} completed in {duration:.2f}s", step=step, rank=rank)


# Global loggers cache
_loggers: dict[str, TelescopeLogger] = {}
_initialized: bool = False
_logs_path: Path | None = None
_debug_mode: bool = False


@dataclass
class LogRecord:
    """A single log record captured for upload."""
    timestamp: float
    level: str
    component: str
    source: str  # "log", "detailed", "stdout"
    message: str


class BufferedLogHandler(logging.Handler):
    """Captures log records in a thread-safe buffer for upload to wandb."""

    def __init__(self, component: str, source: str):
        super().__init__()
        self._component = component
        self._source = source
        self._buffer: list[LogRecord] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        log_record = LogRecord(
            timestamp=record.created,
            level=record.levelname,
            component=self._component,
            source=self._source,
            message=self.format(record),
        )
        with self._lock:
            self._buffer.append(log_record)

    def drain(self) -> list[LogRecord]:
        """Atomically drain and return all buffered records."""
        with self._lock:
            records = self._buffer
            self._buffer = []
            return records


class StdoutTailer:
    """Reads new content from stdout log files by tracking file offsets."""

    def __init__(self, directory: Path, component: str):
        self._directory = directory
        self._component = component
        self._offsets: dict[Path, int] = {}

    def read_new_content(self) -> list[LogRecord]:
        """Read any new content appended to stdout files since last call."""
        records: list[LogRecord] = []
        if not self._directory.exists():
            return records
        for filepath in sorted(self._directory.iterdir()):
            if not filepath.is_file():
                continue
            last_offset = self._offsets.get(filepath, 0)
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(last_offset)
                    new_content = f.read()
                    if new_content:
                        self._offsets[filepath] = f.tell()
                        for line in new_content.splitlines():
                            if line.strip():
                                records.append(LogRecord(
                                    timestamp=time.time(),
                                    level="INFO",
                                    component=self._component,
                                    source="stdout",
                                    message=f"[{filepath.name}] {line}",
                                ))
            except (OSError, IOError):
                continue
        return records


# Registry for buffered log handlers and stdout tailers
_buffered_handlers: dict[str, list[BufferedLogHandler]] = {}
_stdout_tailers: dict[str, StdoutTailer] = {}


def drain_all_log_buffers() -> list[LogRecord]:
    """Drain all buffered log records from all components. Called by EventLogger."""
    all_records: list[LogRecord] = []
    for handlers in _buffered_handlers.values():
        for handler in handlers:
            all_records.extend(handler.drain())
    for tailer in _stdout_tailers.values():
        all_records.extend(tailer.read_new_content())
    return all_records


def setup_stdout_tailers(logs_path: Path):
    """Create stdout tailers for inference and trainer stdout directories."""
    global _stdout_tailers
    stdout_dir = logs_path / "stdout"
    _stdout_tailers["inference"] = StdoutTailer(stdout_dir / "inference", "inference")
    _stdout_tailers["trainer"] = StdoutTailer(stdout_dir / "trainer", "trainer")


def is_debug_mode() -> bool:
    """Return True if Telescope was started with --debug."""
    return _debug_mode


def _resolve_logs_path(logs_dir: Path | str | None) -> Path:
    """Resolve the effective logs directory from args/env defaults."""
    if logs_dir is None:
        run_dir = os.environ.get("TELESCOPE_RUN_DIR")
        if run_dir:
            return (Path(run_dir) / "logs").resolve()
        return Path("logs")
    return Path(logs_dir)


def setup_logging(
    logs_dir: Path | str | None = None,
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.INFO,
    detailed_level: LogLevel = LogLevel.DEBUG,
    debug: bool = False,
) -> Path:
    """
    Initialize the logging system.
    
    Creates directory structure:
        logs/
        ├── orchestrator/
        │   ├── orchestrator.log        # Standard log
        │   └── orchestrator.detailed.log   # Detailed log with debug info
        ├── trainer/
        │   ├── trainer.log
        │   └── trainer.detailed.log
        └── inference/
            ├── inference.log
            └── inference.detailed.log
    
    Args:
        logs_dir: Base directory for logs
        console_level: Minimum level for console output
        file_level: Minimum level for standard log files
        detailed_level: Minimum level for detailed log files
    
    Returns:
        Path to the logs directory
    """
    global _initialized, _logs_path, _debug_mode

    # Prevent duplicate initialization
    if _initialized:
        return _logs_path or _resolve_logs_path(logs_dir)

    _debug_mode = debug

    logs_path = _resolve_logs_path(logs_dir)
    
    # Create directory structure
    components = ["orchestrator", "trainer", "inference"]
    for component in components:
        (logs_path / component).mkdir(parents=True, exist_ok=True)
    
    # Configure root telescope logger
    root_logger = logging.getLogger("telescope")
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, handlers will filter
    root_logger.handlers.clear()  # Clear any existing handlers
    
    # Console handler (clean format, colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(TelescopeFormatter(use_colors=True, console=True))
    root_logger.addHandler(console_handler)
    
    # Set up file handlers for each component
    for component in components:
        component_logger = logging.getLogger(f"telescope.{component}")
        component_logger.setLevel(logging.DEBUG)
        component_logger.handlers.clear()  # Clear any existing handlers
        component_logger.propagate = True  # Still propagate to root for console
        
        component_dir = logs_path / component
        
        # Standard log file
        standard_handler = logging.FileHandler(
            component_dir / f"{component}.log",
            mode="a",
            encoding="utf-8",
        )
        standard_handler.setLevel(file_level)
        standard_handler.setFormatter(TelescopeFormatter(use_colors=False))
        component_logger.addHandler(standard_handler)
        
        # Detailed log file
        detailed_handler = logging.FileHandler(
            component_dir / f"{component}.detailed.log",
            mode="a",
            encoding="utf-8",
        )
        detailed_handler.setLevel(detailed_level)
        detailed_handler.setFormatter(DetailedFormatter(use_colors=False))
        component_logger.addHandler(detailed_handler)

        # Buffered handlers for wandb upload (records will be drained by EventLogger)
        standard_buffered = BufferedLogHandler(component, "log")
        standard_buffered.setLevel(file_level)
        standard_buffered.setFormatter(TelescopeFormatter(use_colors=False))
        component_logger.addHandler(standard_buffered)

        detailed_buffered = BufferedLogHandler(component, "detailed")
        detailed_buffered.setLevel(detailed_level)
        detailed_buffered.setFormatter(DetailedFormatter(use_colors=False))
        component_logger.addHandler(detailed_buffered)

        _buffered_handlers[component] = [standard_buffered, detailed_buffered]

    _initialized = True
    _logs_path = logs_path
    return logs_path


def get_logger(component: str) -> TelescopeLogger:
    """
    Get a logger for a specific component.
    
    Args:
        component: One of "orchestrator", "trainer", "inference"
    
    Returns:
        TelescopeLogger instance for the component
    
    Example:
        log = get_logger("trainer")
        log.info("Starting training", step=0, rank=0)
        log.metric("loss", 0.5, step=0)
    """
    global _loggers, _initialized
    
    if component not in _loggers:
        # Auto-initialize with defaults if not done yet
        if not _initialized:
            setup_logging()
        
        logger = logging.getLogger(f"telescope.{component}")
        _loggers[component] = TelescopeLogger(logger, component)
    
    return _loggers[component]


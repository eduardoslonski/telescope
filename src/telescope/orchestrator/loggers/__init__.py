"""
Logging module for orchestrator.

All events and metrics are consolidated into unified zip archives in the events/ folder:
- events/tail.zip: Last 60 seconds (orchestrator, trainer, inference, gpu, cpu, vllm, thread_pools parquet files)
- events/block_live.zip: Current 30-minute block
- events/block_*.zip: Finalized 30-minute blocks

Provides:
- WandbLogger: High-level W&B logging manager
- EventLogger: Consolidates all data sources and uploads unified zip archives for frontend UI
- SystemMetricsLogger: Collects GPU/CPU metrics every second (data fed to EventLogger)
- VllmMetricsLogger: Collects vLLM inference metrics every 100ms (data fed to EventLogger)
- ThreadPoolMetricsLogger: Collects thread pool utilization every 2 seconds (data fed to EventLogger)
"""
from telescope.orchestrator.loggers.wandb_logger import WandbLogger
from telescope.orchestrator.loggers.event_logger import EventLogger
from telescope.orchestrator.loggers.system_metrics_logger import SystemMetricsLogger
from telescope.orchestrator.loggers.vllm_metrics_logger import VllmMetricsLogger
from telescope.orchestrator.loggers.thread_pool_metrics_logger import ThreadPoolMetricsLogger

__all__ = ["WandbLogger", "EventLogger", "SystemMetricsLogger", "VllmMetricsLogger", "ThreadPoolMetricsLogger"]


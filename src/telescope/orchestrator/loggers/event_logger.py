"""
Event logging system for training UI visualization.

This module implements a parquet-based event logging system that uploads to W&B
in a way that's easy for the frontend UI to fetch and filter.

All events are stored together in unified zip archives:
- events/tail.zip: Last 60 seconds of all data
  Contains: orchestrator.parquet, trainer.parquet, inference.parquet, gpu.parquet, cpu.parquet, vllm.parquet,
            rollouts_discarded.parquet, rollouts_metrics_discarded.parquet,
            golden_answers_discarded.parquet, info_turns_discarded.parquet,
            sample_tags_discarded.parquet, samples_data_discarded.parquet, prompts_discarded.parquet
- events/block_live.zip: Current 30-minute block with all parquet files
- events/block_*.zip: Finalized 30-minute blocks

Each parquet file in events has a tail_idx column indicating which upload cycle
(0, 1, 2, ...) the event/metric was first included in. This helps the frontend
track incremental updates and avoid processing duplicate data.

IMPORTANT: All filtering is done by tail_idx, not timestamp, to ensure complete tails.
This guarantees that all events with a given tail_idx are always included together,
preventing partial data when the frontend fetches incrementally.

Each zip archive includes a metadata.json file with archive-level metadata:
- min_tail_idx: First complete tail index in the archive
- max_tail_idx: Last complete tail index in the archive
- Additional fields depending on the archive type (block_idx, tail_idx, etc.)

Inference events track request and weight broadcast timings per server:
- event_type: "request" or "weight_broadcast"
- start_time, end_time: Event duration
- server: Server index (0, 1, ...)
- step: Training step (populated for weight_broadcast, null for requests)
- tail_idx: Upload cycle index when first uploaded

Steps data is stored in blocks by step count:
- steps/tail.zip: Last 5 training steps' data (rollouts, rewards, metrics)
- steps/block_live.zip: Current block being built (up to 500 steps)
- steps/block_*.zip: Finalized blocks (every 500 steps)

Run summary is updated every 5 seconds with metadata to help the UI:
- events/current_tail_idx: Current tail upload cycle index
- events/current_block_idx: Which event block we're currently writing to
- events/num_finalized_blocks: How many event block_*.zip files have been written
- events/tail_inference_count: Number of inference events in tail
- steps/current_block_idx: Which step block we're currently writing to
- steps/num_finalized_blocks: How many step block_*.zip files have been written
- steps/last_training_step: Last completed trainer step index (-1 before first completion)
- tail_start_time: Oldest event timestamp in tail
- tail_end_time: Newest event timestamp in tail
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from telescope import table_schema_versions
from telescope.utils.tlog import get_logger
from telescope.utils.tlog.logger import drain_all_log_buffers, LogRecord
from telescope.orchestrator.loggers.system_metrics_logger import (
    SystemMetricsLogger,
    GpuMetricSample,
    CpuMetricSample,
)
from telescope.orchestrator.loggers.vllm_metrics_logger import (
    VllmMetricsLogger,
    VllmMetricSample,
)

_log = get_logger("orchestrator")

_zstd_compressor = zstd.ZstdCompressor(level=3)


def _zstd_compress(text: str) -> bytes:
    return _zstd_compressor.compress(text.encode("utf-8"))


# Orchestrator events schema - instant events only
ORCHESTRATOR_EVENT_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("event_type", pa.string()),
    ("step", pa.int32()),
    ("node_id", pa.int32()),
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this event was first uploaded
    ("group_id", pa.int32()),  # Request group ID (shared by all samples in a group, -1 if N/A)
    ("sample_id", pa.int32()),  # Run-wide unique sample index (-1 if N/A)
])

# Trainer events schema - duration events with rank info and hierarchy
TRAINER_EVENT_SCHEMA = pa.schema([
    ("event_type", pa.string()),
    ("step", pa.int32()),
    ("rank", pa.int32()),
    ("local_rank", pa.int32()),
    ("node_id", pa.int32()),
    ("gpu_index", pa.int32()),  # Physical GPU index on node (join to gpu.parquet via node_id+gpu_index)
    ("start_time", pa.float64()),
    ("end_time", pa.float64()),
    ("parent", pa.string()),  # Parent event path, null for root events
    ("depth", pa.int32()),    # Nesting level: 0 for root, 1 for children, etc.
    ("microbatch", pa.int32()),  # Micro batch index (-1 if not a per-microbatch event)
    ("minibatch", pa.int32()),  # Mini batch index (-1 if not applicable or only 1 minibatch)
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this event was first uploaded
])

# Inference events schema - duration events with server info
INFERENCE_EVENT_SCHEMA = pa.schema([
    ("event_type", pa.string()),
    ("start_time", pa.float64()),
    ("end_time", pa.float64()),
    ("server", pa.int32()),  # Server index (0, 1, ...)
    ("node_id", pa.int32()),
    ("tp_group_id", pa.int32()),
    ("tp_size", pa.int32()),
    ("prompt_tokens", pa.int32()),  # Number of prompt tokens
    ("rollout_tokens", pa.int32()),  # Number of generated tokens
    ("group_id", pa.int32()),  # Request group ID (shared by all rollouts in a group)
    ("sample_id", pa.int32()),  # Run-wide unique sample index for this request
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this event was first uploaded
    # vLLM per-request timing from OpenTelemetry spans (populated when ENABLE_VLLM_TRACING=True)
    ("vllm_request_id", pa.string()),  # vLLM request ID (e.g. "cmpl-abc123"), join key for span data
    ("queue_time", pa.float64()),  # Time request waited in vLLM's queue before scheduling
    ("time_to_first_token", pa.float64()),  # TTFT from vLLM's perspective
    ("prefill_time", pa.float64()),  # Time from scheduling to first token (model prefill)
    ("decode_time", pa.float64()),  # Time from first token to last token
    ("inference_time", pa.float64()),  # prefill + decode (scheduling to last token)
    ("e2e_latency", pa.float64()),  # End-to-end latency inside vLLM
    ("vllm_max_tokens", pa.int32()),  # The max_tokens param for this request in vLLM
    ("is_eval", pa.bool_()),  # Whether this request is an eval request (not training)
    ("is_canceled", pa.bool_()),  # Whether this request was cancelled before completing
    ("compute_reward_time", pa.float64()),  # Time for compute_reward/compute_eval_metrics (seconds)
    ("step", pa.int32()),  # Training step (populated for weight_broadcast, -1 for requests)
    ("off_policy_steps", pa.int32()),  # Number of weight updates since this rollout was dispatched
    ("server_lane", pa.int32()),  # Per-server lane slot for this sample
    ("phase", pa.string()),  # "start" or "end" (empty for backward compat e.g. weight_broadcast)
])

# GPU metrics schema - includes gpu_index column
GPU_METRICS_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("node_id", pa.int32()),
    ("gpu_index", pa.int32()),
    ("rank", pa.int32()),
    ("local_rank", pa.int32()),
    ("source", pa.string()),
    ("metric_name", pa.string()),
    ("value", pa.float64()),
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this metric was first uploaded
])

# CPU/System metrics schema
CPU_METRICS_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("node_id", pa.int32()),
    ("source", pa.string()),
    ("metric_name", pa.string()),
    ("value", pa.float64()),
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this metric was first uploaded
])

# vLLM metrics schema
VLLM_METRICS_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("server", pa.int32()),
    ("node_id", pa.int32()),
    ("tp_group_id", pa.int32()),
    ("tp_size", pa.int32()),
    ("metric_name", pa.string()),
    ("value", pa.float64()),
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this metric was first uploaded
])

# Prompts schema - one row per group (all completions in a group share the same prompt)
PROMPTS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),  # Unique per prompt group
    ("env", pa.string()),  # Environment name (e.g. "countdown", "coding")
    ("prompt", pa.string()),  # The initial prompt text
    ("tokens_prompt", pa.int32()),  # Number of prompt tokens
    ("system_prompt", pa.string()),  # The system message (if any)
    ("tokens_system_prompt", pa.int32()),  # Number of tokens in the system message
])

# Rollouts schema - one row per turn per completion (model rollouts and env responses)
# For single-turn: one row per sample_idx with order=0, type="model"
# For multi-turn: multiple rows per sample_idx tracking the conversation
# Each turn's content is just that turn's text, not accumulated from previous turns
ROLLOUTS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),  # Links to prompts table
    ("sample_idx", pa.int32()),  # Unique per completion across the run
    ("turn_order", pa.int32()),  # 0, 1, 2, ... sequence within this sample
    ("turn_type", pa.string()),  # "model" for model rollouts, or env-provided type
    ("content", pa.binary()),  # The text content for this turn only (zstd-compressed UTF-8)
    ("tokens", pa.int32()),  # Number of tokens for this turn
    ("stop_reason", pa.string()),  # Why rollout stopped (for model turns: "stop", "length", etc.)
    ("environment_response_time", pa.float64()),  # Time in seconds for env_response() call (0.0 for model turns)
])

# Samples data schema - one row per sample with summary information
# Links to prompts table via group_id, to rollouts table via sample_idx
SAMPLES_DATA_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),  # Links to prompts table
    ("sample_idx", pa.int32()),  # Unique per completion across the run
    ("reward", pa.float64()),  # Total reward
    ("advantage", pa.float64()),  # Computed advantage
    ("turns", pa.int32()),  # Number of turns in this sample
    ("total_tokens", pa.int32()),  # Total tokens passed to trainer
    ("raw_string", pa.binary()),  # Decoded raw input passed to trainer (zstd-compressed UTF-8)
    ("compute_reward_time", pa.float64()),  # Time in seconds for compute_reward() call
])

# Rollouts metrics schema - normalized table for flexible per-rollout metrics
# Includes both reward components and other rollout metrics (e.g. char count, word frequency)
# Each row represents one metric for one rollout
# UI can query: run_id (from wandb), step+sample_idx (rollout_id), env, metric_name, value
ROLLOUTS_METRICS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("env", pa.string()),  # Environment name
    ("metric_name", pa.string()),  # Metric name (e.g. "format_reward", "equation_reward", "char_count", "word_freq")
    ("value", pa.float64()),  # Metric value
])

# Golden answers schema - ground truth answers per rollout (completely independent from metrics)
# Each row represents one golden answer key/value for one rollout
GOLDEN_ANSWERS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("env", pa.string()),  # Environment name
    ("key", pa.string()),  # Golden answer key name (e.g. "target", "correct_answer")
    ("value", pa.string()),  # Golden answer value (null if not available)
])

# Sample tags schema - per-sample string tags for filtering
# Each row represents one tag key/value for one sample
SAMPLE_TAGS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("env", pa.string()),  # Environment name
    ("tag_name", pa.string()),  # Tag key (e.g. "style", "task")
    ("tag_value", pa.string()),  # Tag value (e.g. "4 numbers", "coding")
])

STEP_METRICS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("section", pa.string()),
    ("group", pa.string()),
    ("metric", pa.string()),
    ("value", pa.float64()),
])

# Discarded prompts schema - one row per discarded group
PROMPTS_DISCARDED_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),  # When the rollout was discarded
    ("discard_reason", pa.string()),  # Why it was discarded (e.g. "max_async", "zero_advantage")
    ("trainer_step", pa.int32()),  # Trainer step at the time of discarding
    ("inference_step", pa.int32()),  # Inference step (what step this rollout would have been for)
    ("group_id", pa.int32()),  # Unique per prompt group
    ("env", pa.string()),  # Environment name (e.g. "countdown", "coding")
    ("prompt", pa.string()),  # The initial prompt text
    ("tokens_prompt", pa.int32()),  # Number of prompt tokens
    ("system_prompt", pa.string()),  # The system message (if any)
    ("tokens_system_prompt", pa.int32()),  # Number of tokens in the system message
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this was first uploaded
])

# Discarded rollouts schema - one row per turn per discarded completion
# Note: timestamp and discard_reason are in samples_data_discarded, not duplicated here
ROLLOUTS_DISCARDED_SCHEMA = pa.schema([
    ("trainer_step", pa.int32()),  # Trainer step at the time of discarding
    ("inference_step", pa.int32()),  # Inference step (what step this rollout would have been for)
    ("group_id", pa.int32()),  # Links to prompts_discarded table
    ("sample_idx", pa.int32()),
    ("turn_order", pa.int32()),  # 0, 1, 2, ... sequence within this sample
    ("turn_type", pa.string()),  # "model" for model rollouts, or env-provided type
    ("content", pa.binary()),  # The text content for this turn only (zstd-compressed UTF-8)
    ("tokens", pa.int32()),  # Number of tokens for this turn
    ("stop_reason", pa.string()),  # Why rollout stopped (for model turns: "stop", "length", etc.)
    ("environment_response_time", pa.float64()),  # Time in seconds for env_response() call (0.0 for model turns)
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this was first uploaded
])

# Discarded samples data schema - one row per discarded sample with summary information
SAMPLES_DATA_DISCARDED_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),  # When the rollout was discarded
    ("discard_reason", pa.string()),  # Why it was discarded
    ("trainer_step", pa.int32()),  # Trainer step at the time of discarding
    ("inference_step", pa.int32()),  # Inference step (what step this rollout would have been for)
    ("group_id", pa.int32()),  # Links to prompts_discarded table
    ("sample_idx", pa.int32()),  # Unique per completion across the run
    ("reward", pa.float64()),  # Total reward
    ("advantage", pa.float64()),  # Computed advantage
    ("turns", pa.int32()),  # Number of turns in this sample
    ("total_tokens", pa.int32()),  # Total tokens that would have been passed to trainer
    ("raw_string", pa.binary()),  # Decoded raw input that would have been passed to trainer (zstd-compressed UTF-8)
    ("compute_reward_time", pa.float64()),  # Time in seconds for compute_reward() call
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this was first uploaded
])

# Discarded rollouts metrics schema - normalized table for flexible per-rollout metrics
# Includes both reward components and other rollout metrics
# Note: timestamp and discard_reason are in samples_data_discarded, not duplicated here
ROLLOUTS_METRICS_DISCARDED_SCHEMA = pa.schema([
    ("sample_idx", pa.int32()),
    ("env", pa.string()),  # Environment name
    ("metric_name", pa.string()),  # Metric name (reward component or other metric)
    ("value", pa.float64()),  # Metric value
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this was first uploaded
])

# Discarded golden answers schema - ground truth answers per discarded rollout
# Note: timestamp and discard_reason are in samples_data_discarded, not duplicated here
GOLDEN_ANSWERS_DISCARDED_SCHEMA = pa.schema([
    ("sample_idx", pa.int32()),
    ("env", pa.string()),  # Environment name
    ("key", pa.string()),  # Golden answer key name
    ("value", pa.string()),  # Golden answer value (null if not available)
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this was first uploaded
])

# Info turns schema - per-turn text information for a sample
# One row per info item. Multiple info items can exist per turn.
# Example uses: stderr from code execution, model summaries, debug output, etc.
INFO_TURNS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("turn_order", pa.int32()),  # Which turn this info is for (0, 1, 2, ...)
    ("env", pa.string()),  # Environment name
    ("info_key", pa.string()),  # Key name (e.g. "stderr", "summary", "debug")
    ("info_value", pa.string()),  # The text value
    ("info_type", pa.string()),  # Type hint for rendering (e.g. "text", "stderr", "stdout")
])

# Discarded info turns schema - same as info_turns but for discarded samples
INFO_TURNS_DISCARDED_SCHEMA = pa.schema([
    ("sample_idx", pa.int32()),
    ("turn_order", pa.int32()),
    ("env", pa.string()),
    ("info_key", pa.string()),
    ("info_value", pa.string()),
    ("info_type", pa.string()),
    ("tail_idx", pa.int32()),  # Index of the tail upload cycle when this was first uploaded
])

# Discarded sample tags schema - per-sample string tags for discarded samples
SAMPLE_TAGS_DISCARDED_SCHEMA = pa.schema([
    ("sample_idx", pa.int32()),
    ("env", pa.string()),
    ("tag_name", pa.string()),
    ("tag_value", pa.string()),
    ("tail_idx", pa.int32()),
])

# ---- Eval schemas (parallel to discarded, for eval completions) ----

PROMPTS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),  # Training step that triggered this eval (0 for baseline, 1-indexed)
    ("eval_name", pa.string()),  # Name of the eval
    ("model_step", pa.int32()),  # Model weights step used for the eval
    ("sample_idx", pa.int32()),  # Index in the eval dataset
    ("env", pa.string()),
    ("prompt", pa.string()),
    ("tokens_prompt", pa.int32()),
    ("system_prompt", pa.string()),
    ("tokens_system_prompt", pa.int32()),
    ("tail_idx", pa.int32()),
])

ROLLOUTS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("model_step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("completion_idx", pa.int32()),  # Which completion for this prompt (0..max(pass_k)-1)
    ("turn_order", pa.int32()),
    ("turn_type", pa.string()),
    ("content", pa.binary()),  # zstd-compressed UTF-8
    ("tokens", pa.int32()),
    ("stop_reason", pa.string()),
    ("environment_response_time", pa.float64()),
    ("tail_idx", pa.int32()),
])

SAMPLES_DATA_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("model_step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("completion_idx", pa.int32()),
    ("env", pa.string()),
    ("turns", pa.int32()),
    ("compute_eval_metrics_time", pa.float64()),
    ("tail_idx", pa.int32()),
])

ROLLOUTS_METRICS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("sample_idx", pa.int32()),
    ("completion_idx", pa.int32()),
    ("env", pa.string()),
    ("metric_name", pa.string()),
    ("value", pa.float64()),
    ("tail_idx", pa.int32()),
])

GOLDEN_ANSWERS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("sample_idx", pa.int32()),
    ("completion_idx", pa.int32()),
    ("env", pa.string()),
    ("key", pa.string()),
    ("value", pa.string()),
    ("tail_idx", pa.int32()),
])

INFO_TURNS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("sample_idx", pa.int32()),
    ("completion_idx", pa.int32()),
    ("turn_order", pa.int32()),
    ("env", pa.string()),
    ("info_key", pa.string()),
    ("info_value", pa.string()),
    ("info_type", pa.string()),
    ("tail_idx", pa.int32()),
])

SAMPLE_TAGS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("sample_idx", pa.int32()),
    ("completion_idx", pa.int32()),
    ("env", pa.string()),
    ("tag_name", pa.string()),
    ("tag_value", pa.string()),
    ("tail_idx", pa.int32()),
])

# Logs schema - captured log records for UI display
LOGS_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("level", pa.string()),
    ("component", pa.string()),
    ("source", pa.string()),
    ("message", pa.string()),
    ("tail_idx", pa.int32()),
])


@dataclass
class Event:
    """Single event in the timeline."""
    timestamp: float
    event_type: str
    source: str  # "trainer" or "orchestrator"
    step: int = -1
    rank: int = -1
    local_rank: int = -1
    node_id: int = -1
    gpu_index: int = -1  # Physical GPU index on node (join to gpu.parquet via node_id+gpu_index)
    node_ip: str = ""
    hostname: str = ""
    ray_node_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    parent: str | None = None  # Parent event path for hierarchy
    depth: int = 0  # Nesting level: 0 for root, 1 for children, etc.
    microbatch: int = -1  # Micro batch index (-1 if not a per-microbatch event)
    minibatch: int = -1  # Mini batch index (-1 if not applicable or only 1 minibatch)
    tail_idx: int = -1  # Index of the tail upload cycle when this event was first uploaded
    group_id: int = -1  # Request group ID (for orchestrator events that relate to a specific group)
    sample_id: int = -1  # Run-wide unique sample index (for orchestrator events that relate to a specific sample)


@dataclass
class RolloutTurn:
    """Single turn in a rollout (model output or env response)."""
    turn_order: int  # 0, 1, 2, ... sequence within this sample
    turn_type: str  # "model" for model rollouts, or env-provided type
    content: str  # The text content for this turn
    tokens: int = 0  # Number of tokens for this turn
    stop_reason: str = ""  # Why rollout stopped (for model turns: "stop", "length", etc.)
    environment_response_time: float = 0.0  # Time in seconds for env_response() call (0.0 for model turns)


@dataclass
class Prompt:
    """Prompt data for a rollout group."""
    step: int
    group_id: int  # Unique per prompt group
    env: str  # Environment name
    prompt: str  # The initial prompt text
    tokens_prompt: int = 0  # Number of prompt tokens
    system_prompt: str = ""  # The system message (if any)
    tokens_system_prompt: int = 0  # Number of tokens in the system message


@dataclass
class Rollout:
    """Single rollout sample with all its turns."""
    step: int
    group_id: int  # Links to prompts table
    sample_idx: int
    env: str  # Environment name
    turns: list[RolloutTurn]  # All turns for this sample (model + env responses)
    reward: float  # Total reward
    advantage: float
    sample_metrics: dict[str, float]  # Per-sample metrics: reward components + any other metrics
    golden_answers: dict[str, str | None] = field(default_factory=dict)  # Golden answer per reward component
    info_turns: list[dict] = field(default_factory=list)  # Per-turn text info (e.g. stderr, summaries)
    sample_tags: dict[str, str] = field(default_factory=dict)  # Per-sample string tags for filtering
    total_tokens: int = 0  # Total tokens passed to trainer
    raw_string: str = ""  # Decoded raw input passed to trainer
    compute_reward_time: float = 0.0  # Time in seconds for compute_reward() call


@dataclass
class StepMetric:
    """Single per-step metric."""
    step: int
    metric: str
    value: float
    section: str = ""
    group: str = ""


@dataclass
class DiscardedPrompt:
    """Prompt data for a discarded rollout group."""
    timestamp: float  # When the rollout was discarded
    discard_reason: str  # Why it was discarded
    trainer_step: int  # Trainer step at the time of discarding
    inference_step: int  # Inference step (what step this rollout would have been for)
    group_id: int  # Unique per prompt group
    env: str  # Environment name
    prompt: str  # The initial prompt text
    tokens_prompt: int = 0  # Number of prompt tokens
    system_prompt: str = ""  # The system message (if any)
    tokens_system_prompt: int = 0  # Number of tokens in the system message
    tail_idx: int = -1  # Index of the tail upload cycle when first uploaded


@dataclass
class DiscardedRollout:
    """A rollout sample that was discarded (not sent to trainer)."""
    timestamp: float  # When the rollout was discarded
    discard_reason: str  # Why it was discarded (e.g. "max_async", "zero_advantage")
    trainer_step: int  # Trainer step at the time of discarding
    inference_step: int  # Inference step (what step this rollout would have been for)
    group_id: int  # Links to prompts_discarded table
    sample_idx: int
    env: str  # Environment name
    turns: list[RolloutTurn]  # All turns for this sample (model + env responses)
    reward: float  # Total reward
    advantage: float
    sample_metrics: dict[str, float]  # Per-sample metrics: reward components + any other metrics
    golden_answers: dict[str, str | None] = field(default_factory=dict)  # Golden answer per reward component
    info_turns: list[dict] = field(default_factory=list)  # Per-turn text info (e.g. stderr, summaries)
    sample_tags: dict[str, str] = field(default_factory=dict)  # Per-sample string tags for filtering
    total_tokens: int = 0  # Total tokens that would have been passed to trainer
    raw_string: str = ""  # Decoded raw input that would have been passed to trainer
    compute_reward_time: float = 0.0  # Time in seconds for compute_reward() call
    tail_idx: int = -1  # Index of the tail upload cycle when first uploaded


@dataclass
class InferenceEvent:
    """Single inference event."""
    event_type: str
    start_time: float
    end_time: float
    server: int  # Server index (0, 1, ...)
    node_id: int = -1
    node_ip: str = ""
    hostname: str = ""
    ray_node_id: str = ""
    tp_group_id: int = -1
    tp_size: int = 1
    prompt_tokens: int = 0  # Number of prompt tokens
    rollout_tokens: int = 0  # Number of generated tokens
    group_id: int = -1  # Request group ID (shared by all rollouts in a group)
    sample_id: int = -1  # Run-wide unique sample index for this request
    tail_idx: int = -1  # Index of the tail upload cycle when this event was first uploaded
    # vLLM per-request timing from OpenTelemetry spans
    vllm_request_id: str = ""  # vLLM request ID (e.g. "cmpl-abc123")
    queue_time: float = 0.0  # Time in vLLM queue
    time_to_first_token: float = 0.0  # TTFT from vLLM
    prefill_time: float = 0.0  # Model prefill time
    decode_time: float = 0.0  # Model decode time
    inference_time: float = 0.0  # prefill + decode
    e2e_latency: float = 0.0  # End-to-end latency inside vLLM
    vllm_max_tokens: int = 0  # max_tokens param for this request
    is_eval: bool = False  # Whether this is an eval request
    is_canceled: bool = False  # Whether this request was cancelled before completing
    compute_reward_time: float = 0.0  # Time for compute_reward/compute_eval_metrics
    step: int = -1  # Training step (populated for weight_broadcast, -1 for requests)
    off_policy_steps: int = 0  # Number of weight updates since this rollout was dispatched
    server_lane: int = -1  # Per-server lane slot for this sample
    phase: str = ""  # "start" or "end" (empty for backward compat e.g. weight_broadcast)


@dataclass
class EvalPrompt:
    """Prompt data for an eval completion group."""
    step: int  # Training step that triggered this eval (0 for baseline, 1-indexed)
    eval_name: str
    model_step: int
    sample_idx: int  # Index in the eval dataset
    env: str
    prompt: str
    tokens_prompt: int = 0
    system_prompt: str = ""
    tokens_system_prompt: int = 0
    tail_idx: int = -1


@dataclass
class EvalRollout:
    """A single eval completion with all its turns."""
    step: int
    eval_name: str
    model_step: int
    sample_idx: int
    completion_idx: int  # 0..max(pass_k)-1
    env: str
    turns: list[RolloutTurn]
    sample_metrics: dict[str, float] = field(default_factory=dict)
    golden_answers: dict[str, str | None] = field(default_factory=dict)
    info_turns: list[dict] = field(default_factory=list)
    sample_tags: dict[str, str] = field(default_factory=dict)
    compute_eval_metrics_time: float = 0.0
    tail_idx: int = -1


class EventLogger:
    """
    Thread-safe event logger that buffers events and periodically uploads to W&B.
    
    This logger consolidates data from multiple sources:
    - Orchestrator events (instant events)
    - Trainer events (duration events with hierarchy)
    - Inference events (request and weight broadcast timings)
    - GPU metrics (from SystemMetricsLogger)
    - CPU metrics (from SystemMetricsLogger)
    - vLLM metrics (from VllmMetricsLogger)
    
    All data is written to unified zip archives in the events/ folder.
    
    Usage:
        logger = EventLogger()
        logger.initialize(wandb_run)
        logger.set_metrics_loggers(system_metrics_logger, vllm_metrics_logger)
        
        # Log events
        logger.log_event("forward", source="trainer", step=0, rank=0, 
                        start_time=t0, end_time=t1)
        logger.log_instant_event("inference_call_start", source="orchestrator")
        
        # Log rollouts
        logger.log_rollout(step=0, sample_idx=0, prompt="...", completion="...", ...)
        
        # Start background upload loop (call once)
        await logger.start_upload_loop()
        
        # Or manually trigger upload
        logger.upload_now()
    """

    @property
    def TAIL_WINDOW_SECONDS(self):
        from telescope.utils import config
        return config.cfg.event_tail_window_seconds

    @property
    def BLOCK_DURATION_SECONDS(self):
        from telescope.utils import config
        return config.cfg.event_block_duration_seconds

    @property
    def ROLLOUT_BLOCK_SIZE(self):
        from telescope.utils import config
        return config.cfg.rollout_block_size

    @property
    def UPLOAD_INTERVAL_SECONDS(self):
        from telescope.utils import config
        return config.cfg.event_upload_interval_seconds

    def __init__(self):
        self.run = None
        self._lock = threading.Lock()

        # Event buffers
        self._events: list[Event] = []
        self._inference_events: list[InferenceEvent] = []
        self._inflight_generations: dict[int, InferenceEvent] = {}  # sample_id -> start event
        self._ended_generation_info: dict[int, InferenceEvent] = {}  # sample_id -> start event (stashed after phase="end")
        self._inflight_compute_reward: dict[int, dict] = {}  # sample_id -> {group_id, server, ...}

        # External metrics loggers (set via set_metrics_loggers)
        self._system_metrics_logger: SystemMetricsLogger | None = None
        self._vllm_metrics_logger: VllmMetricsLogger | None = None
        
        # Accumulated metrics from external loggers (persisted across uploads for block management)
        # Each metric is stored as (metric, tail_idx) tuple
        self._gpu_metrics: list[tuple[GpuMetricSample, int]] = []
        self._cpu_metrics: list[tuple[CpuMetricSample, int]] = []
        self._vllm_metrics: list[tuple[VllmMetricSample, int]] = []

        # Tail tracking - each upload cycle increments the tail_idx
        self._current_tail_idx: int = 0

        # Block tracking
        self._block_start_time: float | None = None
        self._block_first_tail_idx: int = 0  # First tail_idx in the current block
        self._current_block_idx: int = 0
        self._num_finalized_blocks: int = 0

        # Rollout tracking
        self._last_training_step: int = -1
        self._trainer_steps_done: int = 0
        self._pending_rollouts: dict[int, list[Rollout]] = {}
        self._pending_prompts: dict[int, list[Prompt]] = {}  # Prompts per step
        self._logged_group_ids: set[int] = set()  # Track which group_ids have been logged
        
        # Step metrics tracking (grad_norm, kl_divergence_inference, entropy per rank)
        self._pending_step_metrics: dict[int, list[StepMetric]] = {}
        
        # Step block tracking (by step count, not time) - includes rollouts, prompts, and metrics
        self._rollout_block_data: dict[int, list[Rollout]] = {}  # All rollouts in current block
        self._prompts_block_data: dict[int, list[Prompt]] = {}  # All prompts in current block
        self._step_metrics_block_data: dict[int, list[StepMetric]] = {}  # All step metrics in current block
        self._step_block_start_step: int = 0  # First step in current block
        self._current_step_block_idx: int = 0
        self._num_finalized_step_blocks: int = 0

        # Discarded rollouts tracking (uploaded in events zip, not rollouts)
        # These are rollouts that were not sent to the trainer (async limit, zero advantage, etc.)
        self._discarded_rollouts: list[DiscardedRollout] = []
        self._discarded_prompts: list[DiscardedPrompt] = []
        self._logged_discarded_group_ids: set[int] = set()  # Track which discarded group_ids have been logged

        # Eval tracking (parallel to discarded, in events zip)
        self._eval_rollouts: list[EvalRollout] = []
        self._eval_prompts: list[EvalPrompt] = []
        self._logged_eval_prompt_keys: set[tuple] = set()  # (step, eval_name, sample_idx)

        # Log records buffer (drained from logging handlers each upload cycle)
        self._log_records: list[tuple[LogRecord, int]] = []

        # Summary tracking
        self._summary_id: int = 0

        # Upload tracking
        self._last_upload_time: float = 0
        self._upload_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

        # Background thread pool for non-blocking uploads
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="event_logger"
        )
        self._pending_upload: concurrent.futures.Future | None = None

        # Reference time for the run
        self._run_start_time: float = time.time()

    def initialize(self, wandb_run):
        """Initialize with a W&B run object."""
        self.run = wandb_run
        self._run_start_time = time.time()
        self._block_start_time = self._run_start_time
        self._block_first_tail_idx = 0  # First block starts at tail_idx 0
        _log.info(f"EventLogger initialized with run {wandb_run.name if wandb_run else 'None'}")

    def set_metrics_loggers(
        self,
        system_metrics_logger: SystemMetricsLogger | None = None,
        vllm_metrics_logger: VllmMetricsLogger | None = None,
    ):
        """
        Set references to external metrics loggers.
        
        EventLogger will pull metrics from these loggers during upload cycles.
        
        Args:
            system_metrics_logger: SystemMetricsLogger instance for GPU/CPU metrics
            vllm_metrics_logger: VllmMetricsLogger instance for vLLM metrics
        """
        self._system_metrics_logger = system_metrics_logger
        self._vllm_metrics_logger = vllm_metrics_logger
        _log.debug(f"Set metrics loggers: system={system_metrics_logger is not None}, vllm={vllm_metrics_logger is not None}")

    def log_event(
        self,
        event_type: str,
        source: str,
        step: int = -1,
        rank: int = -1,
        local_rank: int = -1,
        node_id: int = -1,
        gpu_index: int = -1,
        node_ip: str = "",
        hostname: str = "",
        ray_node_id: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
        parent: str | None = None,
        depth: int = 0,
        microbatch: int = -1,
        minibatch: int = -1,
    ):
        """Log a duration event (e.g., forward pass, backward pass)."""
        now = time.time()
        start = start_time if start_time is not None else now
        end = end_time if end_time is not None else start

        event = Event(
            timestamp=start,
            event_type=event_type,
            source=source,
            step=step,
            rank=rank,
            local_rank=local_rank,
            node_id=node_id,
            gpu_index=gpu_index,
            node_ip=node_ip,
            hostname=hostname,
            ray_node_id=ray_node_id,
            start_time=start,
            end_time=end,
            parent=parent,
            depth=depth,
            microbatch=microbatch,
            minibatch=minibatch,
        )

        with self._lock:
            self._events.append(event)

    def log_instant_event(
        self,
        event_type: str,
        source: str,
        step: int = -1,
        rank: int = -1,
        local_rank: int = -1,
        node_id: int = -1,
        node_ip: str = "",
        hostname: str = "",
        ray_node_id: str = "",
        timestamp: float | None = None,
        group_id: int = -1,
        sample_id: int = -1,
    ):
        """Log an instant event (e.g., weight update signal)."""
        ts = timestamp if timestamp is not None else time.time()

        event = Event(
            timestamp=ts,
            event_type=event_type,
            source=source,
            step=step,
            rank=rank,
            local_rank=local_rank,
            node_id=node_id,
            node_ip=node_ip,
            hostname=hostname,
            ray_node_id=ray_node_id,
            start_time=ts,
            end_time=ts,
            group_id=group_id,
            sample_id=sample_id,
        )

        with self._lock:
            self._events.append(event)
            # Track inflight compute_reward for snapshot
            if event_type == "compute_reward_start" and sample_id >= 0:
                # Look up server/lane info from the ended generation (stashed after phase="end")
                gen_event = self._ended_generation_info.get(sample_id)
                self._inflight_compute_reward[sample_id] = {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "server": gen_event.server if gen_event else -1,
                    "server_lane": gen_event.server_lane if gen_event else -1,
                    "start_time": gen_event.start_time if gen_event else ts,
                    "is_eval": gen_event.is_eval if gen_event else False,
                    "prompt_tokens": gen_event.prompt_tokens if gen_event else 0,
                }
            elif event_type == "compute_reward_end" and sample_id >= 0:
                self._inflight_compute_reward.pop(sample_id, None)
                self._ended_generation_info.pop(sample_id, None)

    def log_rollout(
        self,
        step: int,
        group_id: int,
        sample_idx: int,
        prompt: str,
        turns: list[dict] | None,
        reward: float,
        advantage: float,
        env: str = "",
        sample_metrics: dict[str, float] | None = None,
        golden_answers: dict[str, str | None] | None = None,
        info_turns: list[dict] | None = None,
        sample_tags: dict[str, str] | None = None,
        tokens_prompt: int = 0,
        system_prompt: str = "",
        tokens_system_prompt: int = 0,
        total_tokens: int = 0,
        raw_string: str = "",
        compute_reward_time: float = 0.0,
    ):
        """
        Log a rollout sample for a training step.

        Args:
            step: Training step
            group_id: Request group ID (shared by all completions with same prompt)
            sample_idx: Run-wide unique sample index (starts at 0, grows throughout run)
            prompt: Input prompt text (for the prompts table)
            turns: List of turn dicts with keys:
                   - "turn_order": int (0, 1, 2, ...)
                   - "turn_type": str ("model" or env-provided type like "tool_result")
                   - "content": str (the text content)
                   - "tokens": int (number of tokens for this turn)
            reward: Total reward
            advantage: Computed advantage
            env: Environment name (e.g. "countdown", "coding")
            sample_metrics: Dict of per-sample metric names to values
                           (e.g. {"format_reward": 1.0, "equation_reward": 0.5, "char_count": 150.0})
            golden_answers: Dict mapping golden answer keys to their values
                           (e.g. {"correct": "42"})
            info_turns: List of per-turn text info dicts with keys:
                       - "turn_order": int (which turn this info is for)
                       - "info_key": str (e.g. "stderr", "summary")
                       - "info_value": str (the text content)
                       - "info_type": str (default "text", can be "stderr", "stdout", etc.)
            sample_tags: Dict of per-sample string tags for filtering
                        (e.g. {"style": "4 numbers", "task": "coding"})
            tokens_prompt: Number of prompt tokens (stored in prompts table)
            system_prompt: The system message (if any)
            tokens_system_prompt: Number of tokens in the system message
            total_tokens: Total tokens passed to trainer (prompt + completion)
            raw_string: Decoded raw input passed to trainer
        """
        # Convert turns to RolloutTurn objects
        if turns is None:
            rollout_turns = []
        else:
            rollout_turns = [
                RolloutTurn(
                    turn_order=t["turn_order"],
                    turn_type=t["turn_type"],
                    content=t["content"],
                    tokens=t.get("tokens", 0),
                    stop_reason=t.get("stop_reason", ""),
                    environment_response_time=t.get("environment_response_time", 0.0),
                )
                for t in turns
            ]
        
        gen = Rollout(
            step=step,
            group_id=group_id,
            sample_idx=sample_idx,
            env=env,
            turns=rollout_turns,
            reward=reward,
            advantage=advantage,
            sample_metrics=sample_metrics or {},
            golden_answers=golden_answers or {},
            info_turns=info_turns or [],
            sample_tags=sample_tags or {},
            total_tokens=total_tokens,
            raw_string=raw_string,
            compute_reward_time=compute_reward_time,
        )

        with self._lock:
            if step not in self._pending_rollouts:
                self._pending_rollouts[step] = []
            self._pending_rollouts[step].append(gen)
            
            # Log prompt if this group_id hasn't been logged yet
            if group_id not in self._logged_group_ids:
                self._logged_group_ids.add(group_id)
                if step not in self._pending_prompts:
                    self._pending_prompts[step] = []
                self._pending_prompts[step].append(Prompt(
                    step=step,
                    group_id=group_id,
                    env=env,
                    prompt=prompt,
                    tokens_prompt=tokens_prompt,
                    system_prompt=system_prompt,
                    tokens_system_prompt=tokens_system_prompt,
                ))
            self._last_training_step = max(self._last_training_step, step)

    def log_inference_event(
        self,
        event_type: str,
        server: int,
        start_time: float,
        end_time: float,
        node_id: int = -1,
        node_ip: str = "",
        hostname: str = "",
        ray_node_id: str = "",
        tp_group_id: int = -1,
        tp_size: int = 1,
        prompt_tokens: int = 0,
        rollout_tokens: int = 0,
        group_id: int = -1,
        sample_id: int = -1,
        vllm_request_id: str = "",
        queue_time: float = 0.0,
        time_to_first_token: float = 0.0,
        prefill_time: float = 0.0,
        decode_time: float = 0.0,
        inference_time: float = 0.0,
        e2e_latency: float = 0.0,
        vllm_max_tokens: int = 0,
        is_eval: bool = False,
        is_canceled: bool = False,
        compute_reward_time: float = 0.0,
        step: int = -1,
        off_policy_steps: int = 0,
        server_lane: int = -1,
        phase: str = "",
    ):
        """
        Log an inference event (request or weight broadcast).

        Args:
            event_type: "request" or "weight_broadcast"
            server: Server index (0, 1, ...)
            start_time: Event start timestamp
            end_time: Event end timestamp
            prompt_tokens: Number of prompt tokens (for requests)
            rollout_tokens: Number of generated tokens (for requests)
            group_id: Request group ID (shared by all rollouts in a group)
            sample_id: Run-wide unique sample index (requests only)
            vllm_request_id: vLLM request ID from completion response (e.g. "cmpl-abc123")
            queue_time: Time request waited in vLLM's queue (from OTLP span)
            time_to_first_token: TTFT from vLLM (from OTLP span)
            prefill_time: Model prefill time (from OTLP span)
            decode_time: Model decode time (from OTLP span)
            inference_time: prefill + decode (from OTLP span)
            e2e_latency: End-to-end latency inside vLLM (from OTLP span)
            vllm_max_tokens: max_tokens param for this request
            is_eval: Whether this is an eval request (not training)
            is_canceled: Whether this request was cancelled before completing
            compute_reward_time: Time for compute_reward or compute_eval_metrics (seconds)
            step: Training step (populated for weight_broadcast events, -1 for requests)
            off_policy_steps: Number of weight updates since this rollout was dispatched
            server_lane: Per-server lane slot for this sample
            phase: "start" or "end" (empty for weight_broadcast events)
        """
        event = InferenceEvent(
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            server=server,
            node_id=node_id,
            node_ip=node_ip,
            hostname=hostname,
            ray_node_id=ray_node_id,
            tp_group_id=tp_group_id,
            tp_size=tp_size,
            prompt_tokens=prompt_tokens,
            rollout_tokens=rollout_tokens,
            group_id=group_id,
            sample_id=sample_id,
            vllm_request_id=vllm_request_id,
            queue_time=queue_time,
            time_to_first_token=time_to_first_token,
            prefill_time=prefill_time,
            decode_time=decode_time,
            inference_time=inference_time,
            e2e_latency=e2e_latency,
            vllm_max_tokens=vllm_max_tokens,
            is_eval=is_eval,
            is_canceled=is_canceled,
            compute_reward_time=compute_reward_time,
            step=step,
            off_policy_steps=off_policy_steps,
            server_lane=server_lane,
            phase=phase,
        )

        with self._lock:
            self._inference_events.append(event)
            # Track inflight generations for snapshot
            if phase == "start" and sample_id >= 0:
                self._inflight_generations[sample_id] = event
            elif phase == "end" and sample_id >= 0:
                # Stash server/lane info for compute_reward tracking before removing
                start_ev = self._inflight_generations.pop(sample_id, None)
                if start_ev is not None:
                    self._ended_generation_info[sample_id] = start_ev

    def log_step_metric(self, step: int, metric: str, value: float, section: str = "", group: str = ""):
        """Log a per-step metric."""
        step_metric = StepMetric(step=step, metric=metric, value=value, section=section, group=group)

        with self._lock:
            if step not in self._pending_step_metrics:
                self._pending_step_metrics[step] = []
            self._pending_step_metrics[step].append(step_metric)

    def set_trainer_steps_done(self, trainer_steps_done: int):
        """Update trainer completion progress for W&B summary metadata."""
        with self._lock:
            self._trainer_steps_done = max(0, trainer_steps_done)

    def log_discarded_rollout(
        self,
        discard_reason: str,
        trainer_step: int,
        inference_step: int,
        group_id: int,
        sample_idx: int,
        prompt: str,
        turns: list[dict] | None,
        reward: float,
        advantage: float,
        env: str = "",
        sample_metrics: dict[str, float] | None = None,
        golden_answers: dict[str, str | None] | None = None,
        info_turns: list[dict] | None = None,
        sample_tags: dict[str, str] | None = None,
        tokens_prompt: int = 0,
        system_prompt: str = "",
        tokens_system_prompt: int = 0,
        timestamp: float | None = None,
        total_tokens: int = 0,
        raw_string: str = "",
        compute_reward_time: float = 0.0,
    ):
        """
        Log a discarded rollout sample.

        These are rollouts that were not sent to the trainer due to:
        - max_async: Async level exceeded the maximum allowed
        - zero_advantage: All samples in the group had zero advantage

        Args:
            discard_reason: Why the rollout was discarded (e.g. "max_async", "zero_advantage")
            trainer_step: Current trainer step at the time of discarding
            inference_step: Current inference step (what step this rollout would have been for)
            group_id: Request group ID (shared by all completions with same prompt)
            sample_idx: Run-wide unique sample index (starts at 0, grows throughout run)
            prompt: Input prompt text (for the prompts_discarded table)
            turns: List of turn dicts with keys:
                   - "turn_order": int (0, 1, 2, ...)
                   - "turn_type": str ("model" or env-provided type)
                   - "content": str (the text content)
                   - "tokens": int (number of tokens for this turn)
            reward: Total reward
            advantage: Computed advantage
            env: Environment name (e.g. "countdown", "coding")
            sample_metrics: Dict of per-sample metric names to values
            golden_answers: Dict mapping golden answer keys to their values
            info_turns: List of per-turn text info dicts (see log_rollout)
            sample_tags: Dict of per-sample string tags for filtering
            tokens_prompt: Number of prompt tokens (stored in prompts_discarded table)
            system_prompt: The system message (if any)
            tokens_system_prompt: Number of tokens in the system message
            timestamp: When the rollout was discarded (defaults to now)
            total_tokens: Total tokens that would have been passed to trainer
            raw_string: Decoded raw input that would have been passed to trainer
        """
        ts = timestamp if timestamp is not None else time.time()
        
        # Convert turns to RolloutTurn objects
        if turns is None:
            rollout_turns = []
        else:
            rollout_turns = [
                RolloutTurn(
                    turn_order=t["turn_order"],
                    turn_type=t["turn_type"],
                    content=t["content"],
                    tokens=t.get("tokens", 0),
                    stop_reason=t.get("stop_reason", ""),
                    environment_response_time=t.get("environment_response_time", 0.0),
                )
                for t in turns
            ]
        
        gen = DiscardedRollout(
            timestamp=ts,
            discard_reason=discard_reason,
            trainer_step=trainer_step,
            inference_step=inference_step,
            group_id=group_id,
            sample_idx=sample_idx,
            env=env,
            turns=rollout_turns,
            reward=reward,
            advantage=advantage,
            sample_metrics=sample_metrics or {},
            golden_answers=golden_answers or {},
            info_turns=info_turns or [],
            sample_tags=sample_tags or {},
            total_tokens=total_tokens,
            raw_string=raw_string,
            compute_reward_time=compute_reward_time,
        )

        with self._lock:
            self._discarded_rollouts.append(gen)
            
            # Log discarded prompt if this group_id hasn't been logged yet
            if group_id not in self._logged_discarded_group_ids:
                self._logged_discarded_group_ids.add(group_id)
                self._discarded_prompts.append(DiscardedPrompt(
                    timestamp=ts,
                    discard_reason=discard_reason,
                    trainer_step=trainer_step,
                    inference_step=inference_step,
                    group_id=group_id,
                    env=env,
                    prompt=prompt,
                    tokens_prompt=tokens_prompt,
                    system_prompt=system_prompt,
                    tokens_system_prompt=tokens_system_prompt,
                ))

    def log_eval_prompt(self, prompt: EvalPrompt):
        """Buffer an eval prompt for upload (deduplicated by step+eval_name+sample_idx)."""
        key = (prompt.step, prompt.eval_name, prompt.sample_idx)
        with self._lock:
            if key not in self._logged_eval_prompt_keys:
                self._logged_eval_prompt_keys.add(key)
                self._eval_prompts.append(prompt)

    def log_eval_rollout(self, rollout: EvalRollout):
        """Buffer an eval rollout for upload."""
        with self._lock:
            self._eval_rollouts.append(rollout)

    def add_gpu_metrics(self, metrics: list[GpuMetricSample]):
        """
        Add GPU metrics from external sources (e.g., trainer torch memory).
        
        These are added to the same buffer as SystemMetricsLogger GPU metrics
        and will be included in the gpu.parquet file.
        
        Args:
            metrics: List of GpuMetricSample objects to add
        """
        if not metrics:
            return
        # Metrics added externally get tail_idx=-1 until assigned during upload
        self._gpu_metrics.extend((m, -1) for m in metrics)

    def add_cpu_metrics(self, metrics: list[CpuMetricSample]):
        """
        Add CPU metrics from external sources (e.g., Ray node snapshots).
        """
        if not metrics:
            return
        # Metrics added externally get tail_idx=-1 until assigned during upload
        self._cpu_metrics.extend((m, -1) for m in metrics)

    def _orchestrator_events_to_table(self, events: list[Event]) -> pa.Table:
        """Convert list of orchestrator events to PyArrow table."""
        if not events:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "event_type": pa.array([], type=pa.string()),
                "step": pa.array([], type=pa.int32()),
                "node_id": pa.array([], type=pa.int32()),
                "tail_idx": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "sample_id": pa.array([], type=pa.int32()),
            })

        data = {
            "timestamp": [e.timestamp for e in events],
            "event_type": [e.event_type for e in events],
            "step": [e.step for e in events],
            "node_id": [e.node_id for e in events],
            "tail_idx": [e.tail_idx for e in events],
            "group_id": [e.group_id for e in events],
            "sample_id": [e.sample_id for e in events],
        }
        return pa.table(data, schema=ORCHESTRATOR_EVENT_SCHEMA)

    def _trainer_events_to_table(self, events: list[Event]) -> pa.Table:
        """Convert list of trainer events to PyArrow table."""
        if not events:
            return pa.table({
                "event_type": pa.array([], type=pa.string()),
                "step": pa.array([], type=pa.int32()),
                "rank": pa.array([], type=pa.int32()),
                "local_rank": pa.array([], type=pa.int32()),
                "node_id": pa.array([], type=pa.int32()),
                "gpu_index": pa.array([], type=pa.int32()),
                "start_time": pa.array([], type=pa.float64()),
                "end_time": pa.array([], type=pa.float64()),
                "parent": pa.array([], type=pa.string()),
                "depth": pa.array([], type=pa.int32()),
                "microbatch": pa.array([], type=pa.int32()),
                "minibatch": pa.array([], type=pa.int32()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "event_type": [e.event_type for e in events],
            "step": [e.step for e in events],
            "rank": [e.rank for e in events],
            "local_rank": [e.local_rank for e in events],
            "node_id": [e.node_id for e in events],
            "gpu_index": [e.gpu_index for e in events],
            "start_time": [e.start_time if e.start_time else e.timestamp for e in events],
            "end_time": [e.end_time if e.end_time else e.timestamp for e in events],
            "parent": [e.parent for e in events],
            "depth": [e.depth for e in events],
            "microbatch": [e.microbatch for e in events],
            "minibatch": [e.minibatch for e in events],
            "tail_idx": [e.tail_idx for e in events],
        }
        return pa.table(data, schema=TRAINER_EVENT_SCHEMA)

    def _inference_events_to_table(self, events: list[InferenceEvent]) -> pa.Table:
        """Convert list of inference events to PyArrow table."""
        if not events:
            return pa.table({
                "event_type": pa.array([], type=pa.string()),
                "start_time": pa.array([], type=pa.float64()),
                "end_time": pa.array([], type=pa.float64()),
                "server": pa.array([], type=pa.int32()),
                "node_id": pa.array([], type=pa.int32()),
                "tp_group_id": pa.array([], type=pa.int32()),
                "tp_size": pa.array([], type=pa.int32()),
                "prompt_tokens": pa.array([], type=pa.int32()),
                "rollout_tokens": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "sample_id": pa.array([], type=pa.int32()),
                "tail_idx": pa.array([], type=pa.int32()),
                "vllm_request_id": pa.array([], type=pa.string()),
                "queue_time": pa.array([], type=pa.float64()),
                "time_to_first_token": pa.array([], type=pa.float64()),
                "prefill_time": pa.array([], type=pa.float64()),
                "decode_time": pa.array([], type=pa.float64()),
                "inference_time": pa.array([], type=pa.float64()),
                "e2e_latency": pa.array([], type=pa.float64()),
                "vllm_max_tokens": pa.array([], type=pa.int32()),
                "is_eval": pa.array([], type=pa.bool_()),
                "is_canceled": pa.array([], type=pa.bool_()),
                "compute_reward_time": pa.array([], type=pa.float64()),
                "step": pa.array([], type=pa.int32()),
                "off_policy_steps": pa.array([], type=pa.int32()),
                "server_lane": pa.array([], type=pa.int32()),
                "phase": pa.array([], type=pa.string()),
            })

        data = {
            "event_type": [e.event_type for e in events],
            "start_time": [e.start_time for e in events],
            "end_time": [e.end_time for e in events],
            "server": [e.server for e in events],
            "node_id": [e.node_id for e in events],
            "tp_group_id": [e.tp_group_id for e in events],
            "tp_size": [e.tp_size for e in events],
            "prompt_tokens": [e.prompt_tokens for e in events],
            "rollout_tokens": [e.rollout_tokens for e in events],
            "group_id": [e.group_id for e in events],
            "sample_id": [e.sample_id for e in events],
            "tail_idx": [e.tail_idx for e in events],
            "vllm_request_id": [e.vllm_request_id for e in events],
            "queue_time": [e.queue_time for e in events],
            "time_to_first_token": [e.time_to_first_token for e in events],
            "prefill_time": [e.prefill_time for e in events],
            "decode_time": [e.decode_time for e in events],
            "inference_time": [e.inference_time for e in events],
            "e2e_latency": [e.e2e_latency for e in events],
            "vllm_max_tokens": [e.vllm_max_tokens for e in events],
            "is_eval": [e.is_eval for e in events],
            "is_canceled": [e.is_canceled for e in events],
            "compute_reward_time": [e.compute_reward_time for e in events],
            "step": [e.step if e.step != -1 else None for e in events],
            "off_policy_steps": [e.off_policy_steps for e in events],
            "server_lane": [e.server_lane for e in events],
            "phase": [e.phase for e in events],
        }
        return pa.table(data, schema=INFERENCE_EVENT_SCHEMA)

    def _gpu_metrics_to_table(self, metrics: list[tuple[GpuMetricSample, int]]) -> pa.Table:
        """Convert list of GPU metrics (with tail_idx) to PyArrow table."""
        if not metrics:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "node_id": pa.array([], type=pa.int32()),
                "gpu_index": pa.array([], type=pa.int32()),
                "rank": pa.array([], type=pa.int32()),
                "local_rank": pa.array([], type=pa.int32()),
                "source": pa.array([], type=pa.string()),
                "metric_name": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "timestamp": [m.timestamp for m, _ in metrics],
            "node_id": [m.node_id for m, _ in metrics],
            "gpu_index": [m.gpu_index for m, _ in metrics],
            "rank": [m.rank for m, _ in metrics],
            "local_rank": [m.local_rank for m, _ in metrics],
            "source": [m.source for m, _ in metrics],
            "metric_name": [m.metric_name for m, _ in metrics],
            "value": [m.value for m, _ in metrics],
            "tail_idx": [tail_idx for _, tail_idx in metrics],
        }
        return pa.table(data, schema=GPU_METRICS_SCHEMA)

    def _cpu_metrics_to_table(self, metrics: list[tuple[CpuMetricSample, int]]) -> pa.Table:
        """Convert list of CPU metrics (with tail_idx) to PyArrow table."""
        if not metrics:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "node_id": pa.array([], type=pa.int32()),
                "source": pa.array([], type=pa.string()),
                "metric_name": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "timestamp": [m.timestamp for m, _ in metrics],
            "node_id": [m.node_id for m, _ in metrics],
            "source": [m.source for m, _ in metrics],
            "metric_name": [m.metric_name for m, _ in metrics],
            "value": [m.value for m, _ in metrics],
            "tail_idx": [tail_idx for _, tail_idx in metrics],
        }
        return pa.table(data, schema=CPU_METRICS_SCHEMA)

    def _logs_to_table(self, logs: list[tuple[LogRecord, int]]) -> pa.Table:
        """Convert log records (with tail_idx) to PyArrow table."""
        if not logs:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "level": pa.array([], type=pa.string()),
                "component": pa.array([], type=pa.string()),
                "source": pa.array([], type=pa.string()),
                "message": pa.array([], type=pa.string()),
                "tail_idx": pa.array([], type=pa.int32()),
            })
        data = {
            "timestamp": [r.timestamp for r, _ in logs],
            "level": [r.level for r, _ in logs],
            "component": [r.component for r, _ in logs],
            "source": [r.source for r, _ in logs],
            "message": [r.message for r, _ in logs],
            "tail_idx": [tail_idx for _, tail_idx in logs],
        }
        return pa.table(data, schema=LOGS_SCHEMA)

    def _vllm_metrics_to_table(self, metrics: list[tuple[VllmMetricSample, int]]) -> pa.Table:
        """Convert list of vLLM metrics (with tail_idx) to PyArrow table."""
        if not metrics:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "server": pa.array([], type=pa.int32()),
                "node_id": pa.array([], type=pa.int32()),
                "tp_group_id": pa.array([], type=pa.int32()),
                "tp_size": pa.array([], type=pa.int32()),
                "metric_name": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "timestamp": [m.timestamp for m, _ in metrics],
            "server": [m.server for m, _ in metrics],
            "node_id": [m.node_id for m, _ in metrics],
            "tp_group_id": [m.tp_group_id for m, _ in metrics],
            "tp_size": [m.tp_size for m, _ in metrics],
            "metric_name": [m.metric_name for m, _ in metrics],
            "value": [m.value for m, _ in metrics],
            "tail_idx": [tail_idx for _, tail_idx in metrics],
        }
        return pa.table(data, schema=VLLM_METRICS_SCHEMA)

    def _prompts_to_table(self, prompts: list[Prompt]) -> pa.Table:
        """Convert list of prompts to PyArrow table."""
        if not prompts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "prompt": pa.array([], type=pa.string()),
                "tokens_prompt": pa.array([], type=pa.int32()),
                "system_prompt": pa.array([], type=pa.string()),
                "tokens_system_prompt": pa.array([], type=pa.int32()),
            })

        data = {
            "step": [p.step for p in prompts],
            "group_id": [p.group_id for p in prompts],
            "env": [p.env for p in prompts],
            "prompt": [p.prompt for p in prompts],
            "tokens_prompt": [p.tokens_prompt for p in prompts],
            "system_prompt": [p.system_prompt for p in prompts],
            "tokens_system_prompt": [p.tokens_system_prompt for p in prompts],
        }
        return pa.table(data, schema=PROMPTS_SCHEMA)

    def _rollouts_to_table(self, rollouts: list[Rollout]) -> pa.Table:
        """
        Convert list of rollouts to PyArrow table.
        
        Each rollout may have multiple turns, so this creates one row per turn.
        Each turn's content is just that turn's text, not accumulated from previous turns.
        Summary fields (reward, advantage) are only filled on the last turn.
        tokens is filled for every turn.
        """
        if not rollouts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "turn_order": pa.array([], type=pa.int32()),
                "turn_type": pa.array([], type=pa.string()),
                "content": pa.array([], type=pa.binary()),
                "tokens": pa.array([], type=pa.int32()),
                "stop_reason": pa.array([], type=pa.string()),
                "environment_response_time": pa.array([], type=pa.float64()),
            })

        # Flatten rollouts into turn rows
        steps = []
        group_ids = []
        sample_idxs = []
        turn_orders = []
        turn_types = []
        contents = []
        tokens_list = []
        stop_reasons = []
        env_response_times = []

        for gen in rollouts:
            for turn in gen.turns:
                steps.append(gen.step)
                group_ids.append(gen.group_id)
                sample_idxs.append(gen.sample_idx)
                turn_orders.append(turn.turn_order)
                turn_types.append(turn.turn_type)
                contents.append(_zstd_compress(turn.content))
                tokens_list.append(turn.tokens)
                stop_reasons.append(turn.stop_reason)
                env_response_times.append(turn.environment_response_time)

        data = {
            "step": steps,
            "group_id": group_ids,
            "sample_idx": sample_idxs,
            "turn_order": turn_orders,
            "turn_type": turn_types,
            "content": contents,
            "tokens": tokens_list,
            "stop_reason": stop_reasons,
            "environment_response_time": env_response_times,
        }
        return pa.table(data, schema=ROLLOUTS_SCHEMA)

    def _samples_data_to_table(self, rollouts: list[Rollout]) -> pa.Table:
        """
        Convert rollouts to a samples_data table with summary information per sample.
        
        Creates one row per sample with reward, advantage, turn count, total_tokens, raw_string,
        and compute_reward_time.
        """
        if not rollouts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "reward": pa.array([], type=pa.float64()),
                "advantage": pa.array([], type=pa.float64()),
                "turns": pa.array([], type=pa.int32()),
                "total_tokens": pa.array([], type=pa.int32()),
                "raw_string": pa.array([], type=pa.binary()),
                "compute_reward_time": pa.array([], type=pa.float64()),
            })

        data = {
            "step": [gen.step for gen in rollouts],
            "group_id": [gen.group_id for gen in rollouts],
            "sample_idx": [gen.sample_idx for gen in rollouts],
            "reward": [gen.reward for gen in rollouts],
            "advantage": [gen.advantage for gen in rollouts],
            "turns": [len(gen.turns) for gen in rollouts],
            "total_tokens": [gen.total_tokens for gen in rollouts],
            "raw_string": [_zstd_compress(gen.raw_string) for gen in rollouts],
            "compute_reward_time": [gen.compute_reward_time for gen in rollouts],
        }
        return pa.table(data, schema=SAMPLES_DATA_SCHEMA)

    def _rollouts_metrics_to_table(self, rollouts: list[Rollout]) -> pa.Table:
        """
        Convert rollouts' reward components and rollout metrics to a normalized table.
        
        Creates one row per metric per rollout, combining both reward components
        and other rollout metrics (e.g. char count, word frequency).
        
        Min/max ranges for metrics are registered at the environment level
        (Environment.metrics_ranges) and uploaded in env_details, not per-sample.
        """
        if not rollouts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "metric_name": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
            })

        # Explode reward components + rollout metrics into normalized rows
        steps = []
        sample_idxs = []
        envs = []
        metric_names = []
        values = []

        for gen in rollouts:
            for metric_name, value in gen.sample_metrics.items():
                steps.append(gen.step)
                sample_idxs.append(gen.sample_idx)
                envs.append(gen.env)
                metric_names.append(metric_name)
                values.append(float(value))

        data = {
            "step": steps,
            "sample_idx": sample_idxs,
            "env": envs,
            "metric_name": metric_names,
            "value": values,
        }
        return pa.table(data, schema=ROLLOUTS_METRICS_SCHEMA)

    def _golden_answers_to_table(self, rollouts: list[Rollout]) -> pa.Table:
        """
        Convert rollouts' golden answers to a normalized table.
        
        Creates one row per golden answer key/value per rollout.
        Golden answers are completely independent from rollout metrics.
        """
        if not rollouts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "key": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.string()),
            })

        steps = []
        sample_idxs = []
        envs = []
        keys = []
        values = []

        for gen in rollouts:
            for key, value in gen.golden_answers.items():
                steps.append(gen.step)
                sample_idxs.append(gen.sample_idx)
                envs.append(gen.env)
                keys.append(key)
                values.append(value)

        data = {
            "step": steps,
            "sample_idx": sample_idxs,
            "env": envs,
            "key": keys,
            "value": values,
        }
        return pa.table(data, schema=GOLDEN_ANSWERS_SCHEMA)

    def _info_turns_to_table(self, rollouts: list[Rollout]) -> pa.Table:
        """
        Convert rollouts' info_turns to a normalized table.
        
        Creates one row per info item per turn per rollout.
        Each info item has a key, value (text), and type hint for frontend rendering.
        """
        if not rollouts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "turn_order": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "info_key": pa.array([], type=pa.string()),
                "info_value": pa.array([], type=pa.string()),
                "info_type": pa.array([], type=pa.string()),
            })

        steps = []
        sample_idxs = []
        turn_orders = []
        envs = []
        info_keys = []
        info_values = []
        info_types = []

        for gen in rollouts:
            for info in gen.info_turns:
                steps.append(gen.step)
                sample_idxs.append(gen.sample_idx)
                turn_orders.append(info.get("turn_order", 0))
                envs.append(gen.env)
                info_keys.append(info.get("info_key", ""))
                info_values.append(info.get("info_value", ""))
                info_types.append(info.get("info_type", "text"))

        data = {
            "step": steps,
            "sample_idx": sample_idxs,
            "turn_order": turn_orders,
            "env": envs,
            "info_key": info_keys,
            "info_value": info_values,
            "info_type": info_types,
        }
        return pa.table(data, schema=INFO_TURNS_SCHEMA)

    def _sample_tags_to_table(self, rollouts: list[Rollout]) -> pa.Table:
        """Convert rollouts' sample_tags to a normalized table (one row per tag per rollout)."""
        if not rollouts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "tag_name": pa.array([], type=pa.string()),
                "tag_value": pa.array([], type=pa.string()),
            })

        steps = []
        sample_idxs = []
        envs = []
        tag_names = []
        tag_values = []

        for gen in rollouts:
            for tag_name, tag_value in gen.sample_tags.items():
                steps.append(gen.step)
                sample_idxs.append(gen.sample_idx)
                envs.append(gen.env)
                tag_names.append(tag_name)
                tag_values.append(str(tag_value))

        data = {
            "step": steps,
            "sample_idx": sample_idxs,
            "env": envs,
            "tag_name": tag_names,
            "tag_value": tag_values,
        }
        return pa.table(data, schema=SAMPLE_TAGS_SCHEMA)

    def _step_metrics_to_table(self, step_metrics: list[StepMetric]) -> pa.Table:
        """Convert list of step metrics to PyArrow table."""
        if not step_metrics:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "section": pa.array([], type=pa.string()),
                "group": pa.array([], type=pa.string()),
                "metric": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
            })

        data = {
            "step": [m.step for m in step_metrics],
            "section": [m.section for m in step_metrics],
            "group": [m.group for m in step_metrics],
            "metric": [m.metric for m in step_metrics],
            "value": [m.value for m in step_metrics],
        }
        return pa.table(data, schema=STEP_METRICS_SCHEMA)

    def _discarded_prompts_to_table(self, prompts: list[DiscardedPrompt]) -> pa.Table:
        """Convert list of discarded prompts to PyArrow table."""
        if not prompts:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "discard_reason": pa.array([], type=pa.string()),
                "trainer_step": pa.array([], type=pa.int32()),
                "inference_step": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "prompt": pa.array([], type=pa.string()),
                "tokens_prompt": pa.array([], type=pa.int32()),
                "system_prompt": pa.array([], type=pa.string()),
                "tokens_system_prompt": pa.array([], type=pa.int32()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "timestamp": [p.timestamp for p in prompts],
            "discard_reason": [p.discard_reason for p in prompts],
            "trainer_step": [p.trainer_step for p in prompts],
            "inference_step": [p.inference_step for p in prompts],
            "group_id": [p.group_id for p in prompts],
            "env": [p.env for p in prompts],
            "prompt": [p.prompt for p in prompts],
            "tokens_prompt": [p.tokens_prompt for p in prompts],
            "system_prompt": [p.system_prompt for p in prompts],
            "tokens_system_prompt": [p.tokens_system_prompt for p in prompts],
            "tail_idx": [p.tail_idx for p in prompts],
        }
        return pa.table(data, schema=PROMPTS_DISCARDED_SCHEMA)

    def _discarded_rollouts_to_table(self, rollouts: list[DiscardedRollout]) -> pa.Table:
        """
        Convert list of discarded rollouts to PyArrow table.
        
        Each rollout may have multiple turns, so this creates one row per turn.
        tokens is filled for every turn.
        Note: timestamp and discard_reason are in samples_data_discarded, not here.
        """
        if not rollouts:
            return pa.table({
                "trainer_step": pa.array([], type=pa.int32()),
                "inference_step": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "turn_order": pa.array([], type=pa.int32()),
                "turn_type": pa.array([], type=pa.string()),
                "content": pa.array([], type=pa.binary()),
                "tokens": pa.array([], type=pa.int32()),
                "stop_reason": pa.array([], type=pa.string()),
                "environment_response_time": pa.array([], type=pa.float64()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        # Flatten rollouts into turn rows
        trainer_steps = []
        inference_steps = []
        group_ids = []
        sample_idxs = []
        turn_orders = []
        turn_types = []
        contents = []
        tokens_list = []
        stop_reasons = []
        env_response_times = []
        tail_idxs = []

        for gen in rollouts:
            for turn in gen.turns:
                trainer_steps.append(gen.trainer_step)
                inference_steps.append(gen.inference_step)
                group_ids.append(gen.group_id)
                sample_idxs.append(gen.sample_idx)
                turn_orders.append(turn.turn_order)
                turn_types.append(turn.turn_type)
                contents.append(_zstd_compress(turn.content))
                tokens_list.append(turn.tokens)
                stop_reasons.append(turn.stop_reason)
                env_response_times.append(turn.environment_response_time)
                tail_idxs.append(gen.tail_idx)

        data = {
            "trainer_step": trainer_steps,
            "inference_step": inference_steps,
            "group_id": group_ids,
            "sample_idx": sample_idxs,
            "turn_order": turn_orders,
            "turn_type": turn_types,
            "content": contents,
            "tokens": tokens_list,
            "stop_reason": stop_reasons,
            "environment_response_time": env_response_times,
            "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=ROLLOUTS_DISCARDED_SCHEMA)

    def _samples_data_discarded_to_table(self, rollouts: list[DiscardedRollout]) -> pa.Table:
        """
        Convert discarded rollouts to a samples_data table with summary information per sample.
        
        Creates one row per discarded sample with reward, advantage, turn count, total_tokens,
        raw_string, and compute_reward_time.
        """
        if not rollouts:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "discard_reason": pa.array([], type=pa.string()),
                "trainer_step": pa.array([], type=pa.int32()),
                "inference_step": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "sample_idx": pa.array([], type=pa.int32()),
                "reward": pa.array([], type=pa.float64()),
                "advantage": pa.array([], type=pa.float64()),
                "turns": pa.array([], type=pa.int32()),
                "total_tokens": pa.array([], type=pa.int32()),
                "raw_string": pa.array([], type=pa.binary()),
                "compute_reward_time": pa.array([], type=pa.float64()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "timestamp": [gen.timestamp for gen in rollouts],
            "discard_reason": [gen.discard_reason for gen in rollouts],
            "trainer_step": [gen.trainer_step for gen in rollouts],
            "inference_step": [gen.inference_step for gen in rollouts],
            "group_id": [gen.group_id for gen in rollouts],
            "sample_idx": [gen.sample_idx for gen in rollouts],
            "reward": [gen.reward for gen in rollouts],
            "advantage": [gen.advantage for gen in rollouts],
            "turns": [len(gen.turns) for gen in rollouts],
            "total_tokens": [gen.total_tokens for gen in rollouts],
            "raw_string": [_zstd_compress(gen.raw_string) for gen in rollouts],
            "compute_reward_time": [gen.compute_reward_time for gen in rollouts],
            "tail_idx": [gen.tail_idx for gen in rollouts],
        }
        return pa.table(data, schema=SAMPLES_DATA_DISCARDED_SCHEMA)

    def _rollouts_metrics_discarded_to_table(self, rollouts: list[DiscardedRollout]) -> pa.Table:
        """
        Convert discarded rollouts' reward components and metrics to a normalized table.
        
        Creates one row per metric per discarded rollout, combining both reward
        components and other rollout metrics.
        Note: timestamp and discard_reason are in samples_data_discarded, not here.
        
        Min/max ranges for metrics are registered at the environment level
        (Environment.metrics_ranges) and uploaded in env_details, not per-sample.
        """
        if not rollouts:
            return pa.table({
                "sample_idx": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "metric_name": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        # Explode sample metrics into normalized rows
        sample_idxs = []
        envs = []
        metric_names = []
        values = []
        tail_idxs = []

        for gen in rollouts:
            for metric_name, value in gen.sample_metrics.items():
                sample_idxs.append(gen.sample_idx)
                envs.append(gen.env)
                metric_names.append(metric_name)
                values.append(float(value))
                tail_idxs.append(gen.tail_idx)

        data = {
            "sample_idx": sample_idxs,
            "env": envs,
            "metric_name": metric_names,
            "value": values,
            "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=ROLLOUTS_METRICS_DISCARDED_SCHEMA)

    def _golden_answers_discarded_to_table(self, rollouts: list[DiscardedRollout]) -> pa.Table:
        """
        Convert discarded rollouts' golden answers to a normalized table.
        
        Creates one row per golden answer key/value per discarded rollout.
        Golden answers are completely independent from rollout metrics.
        Note: timestamp and discard_reason are in samples_data_discarded, not here.
        """
        if not rollouts:
            return pa.table({
                "sample_idx": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "key": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.string()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        sample_idxs = []
        envs = []
        keys = []
        values = []
        tail_idxs = []

        for gen in rollouts:
            for key, value in gen.golden_answers.items():
                sample_idxs.append(gen.sample_idx)
                envs.append(gen.env)
                keys.append(key)
                values.append(value)
                tail_idxs.append(gen.tail_idx)

        data = {
            "sample_idx": sample_idxs,
            "env": envs,
            "key": keys,
            "value": values,
            "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=GOLDEN_ANSWERS_DISCARDED_SCHEMA)

    def _info_turns_discarded_to_table(self, rollouts: list[DiscardedRollout]) -> pa.Table:
        """
        Convert discarded rollouts' info_turns to a normalized table.
        
        Creates one row per info item per turn per discarded rollout.
        Note: timestamp and discard_reason are in samples_data_discarded, not here.
        """
        if not rollouts:
            return pa.table({
                "sample_idx": pa.array([], type=pa.int32()),
                "turn_order": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "info_key": pa.array([], type=pa.string()),
                "info_value": pa.array([], type=pa.string()),
                "info_type": pa.array([], type=pa.string()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        sample_idxs = []
        turn_orders = []
        envs = []
        info_keys = []
        info_values = []
        info_types = []
        tail_idxs = []

        for gen in rollouts:
            for info in gen.info_turns:
                sample_idxs.append(gen.sample_idx)
                turn_orders.append(info.get("turn_order", 0))
                envs.append(gen.env)
                info_keys.append(info.get("info_key", ""))
                info_values.append(info.get("info_value", ""))
                info_types.append(info.get("info_type", "text"))
                tail_idxs.append(gen.tail_idx)

        data = {
            "sample_idx": sample_idxs,
            "turn_order": turn_orders,
            "env": envs,
            "info_key": info_keys,
            "info_value": info_values,
            "info_type": info_types,
            "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=INFO_TURNS_DISCARDED_SCHEMA)

    def _sample_tags_discarded_to_table(self, rollouts: list[DiscardedRollout]) -> pa.Table:
        """Convert discarded rollouts' sample_tags to a normalized table."""
        if not rollouts:
            return pa.table({
                "sample_idx": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "tag_name": pa.array([], type=pa.string()),
                "tag_value": pa.array([], type=pa.string()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        sample_idxs = []
        envs = []
        tag_names = []
        tag_values = []
        tail_idxs = []

        for gen in rollouts:
            for tag_name, tag_value in gen.sample_tags.items():
                sample_idxs.append(gen.sample_idx)
                envs.append(gen.env)
                tag_names.append(tag_name)
                tag_values.append(str(tag_value))
                tail_idxs.append(gen.tail_idx)

        data = {
            "sample_idx": sample_idxs,
            "env": envs,
            "tag_name": tag_names,
            "tag_value": tag_values,
            "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=SAMPLE_TAGS_DISCARDED_SCHEMA)

    # ------------------------------------------------------------------
    # Eval table converters (mirror discarded pattern)
    # ------------------------------------------------------------------

    def _eval_prompts_to_table(self, prompts: list[EvalPrompt]) -> pa.Table:
        if not prompts:
            return pa.table({f.name: pa.array([], type=f.type) for f in PROMPTS_EVAL_SCHEMA})
        data = {
            "step": [p.step for p in prompts],
            "eval_name": [p.eval_name for p in prompts],
            "model_step": [p.model_step for p in prompts],
            "sample_idx": [p.sample_idx for p in prompts],
            "env": [p.env for p in prompts],
            "prompt": [p.prompt for p in prompts],
            "tokens_prompt": [p.tokens_prompt for p in prompts],
            "system_prompt": [p.system_prompt for p in prompts],
            "tokens_system_prompt": [p.tokens_system_prompt for p in prompts],
            "tail_idx": [p.tail_idx for p in prompts],
        }
        return pa.table(data, schema=PROMPTS_EVAL_SCHEMA)

    def _eval_rollouts_to_table(self, rollouts: list[EvalRollout]) -> pa.Table:
        if not rollouts:
            return pa.table({f.name: pa.array([], type=f.type) for f in ROLLOUTS_EVAL_SCHEMA})
        steps, eval_names, model_steps, sample_idxs, comp_idxs = [], [], [], [], []
        turn_orders, turn_types, contents, tokens_list, stop_reasons, env_resp_times, tail_idxs = [], [], [], [], [], [], []
        for r in rollouts:
            for turn in r.turns:
                steps.append(r.step)
                eval_names.append(r.eval_name)
                model_steps.append(r.model_step)
                sample_idxs.append(r.sample_idx)
                comp_idxs.append(r.completion_idx)
                turn_orders.append(turn.turn_order)
                turn_types.append(turn.turn_type)
                contents.append(_zstd_compress(turn.content))
                tokens_list.append(turn.tokens)
                stop_reasons.append(turn.stop_reason)
                env_resp_times.append(turn.environment_response_time)
                tail_idxs.append(r.tail_idx)
        data = {
            "step": steps, "eval_name": eval_names, "model_step": model_steps,
            "sample_idx": sample_idxs, "completion_idx": comp_idxs,
            "turn_order": turn_orders, "turn_type": turn_types, "content": contents,
            "tokens": tokens_list, "stop_reason": stop_reasons,
            "environment_response_time": env_resp_times, "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=ROLLOUTS_EVAL_SCHEMA)

    def _samples_data_eval_to_table(self, rollouts: list[EvalRollout]) -> pa.Table:
        if not rollouts:
            return pa.table({f.name: pa.array([], type=f.type) for f in SAMPLES_DATA_EVAL_SCHEMA})
        data = {
            "step": [r.step for r in rollouts],
            "eval_name": [r.eval_name for r in rollouts],
            "model_step": [r.model_step for r in rollouts],
            "sample_idx": [r.sample_idx for r in rollouts],
            "completion_idx": [r.completion_idx for r in rollouts],
            "env": [r.env for r in rollouts],
            "turns": [len(r.turns) for r in rollouts],
            "compute_eval_metrics_time": [r.compute_eval_metrics_time for r in rollouts],
            "tail_idx": [r.tail_idx for r in rollouts],
        }
        return pa.table(data, schema=SAMPLES_DATA_EVAL_SCHEMA)

    def _rollouts_metrics_eval_to_table(self, rollouts: list[EvalRollout]) -> pa.Table:
        if not rollouts:
            return pa.table({f.name: pa.array([], type=f.type) for f in ROLLOUTS_METRICS_EVAL_SCHEMA})
        steps, eval_names, sample_idxs, comp_idxs, envs, metric_names, values, tail_idxs = [], [], [], [], [], [], [], []
        for r in rollouts:
            for metric_name, value in r.sample_metrics.items():
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                comp_idxs.append(r.completion_idx)
                envs.append(r.env)
                metric_names.append(metric_name)
                values.append(float(value))
                tail_idxs.append(r.tail_idx)
        data = {
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "completion_idx": comp_idxs,
            "env": envs, "metric_name": metric_names,
            "value": values, "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=ROLLOUTS_METRICS_EVAL_SCHEMA)

    def _golden_answers_eval_to_table(self, rollouts: list[EvalRollout]) -> pa.Table:
        if not rollouts:
            return pa.table({f.name: pa.array([], type=f.type) for f in GOLDEN_ANSWERS_EVAL_SCHEMA})
        steps, eval_names, sample_idxs, comp_idxs, envs, keys, values, tail_idxs = [], [], [], [], [], [], [], []
        for r in rollouts:
            for key, value in r.golden_answers.items():
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                comp_idxs.append(r.completion_idx)
                envs.append(r.env)
                keys.append(key)
                values.append(value)
                tail_idxs.append(r.tail_idx)
        data = {
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "completion_idx": comp_idxs,
            "env": envs, "key": keys, "value": values, "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=GOLDEN_ANSWERS_EVAL_SCHEMA)

    def _info_turns_eval_to_table(self, rollouts: list[EvalRollout]) -> pa.Table:
        if not rollouts:
            return pa.table({f.name: pa.array([], type=f.type) for f in INFO_TURNS_EVAL_SCHEMA})
        steps, eval_names, sample_idxs, comp_idxs, turn_orders, envs = [], [], [], [], [], []
        info_keys, info_values, info_types, tail_idxs = [], [], [], []
        for r in rollouts:
            for info in r.info_turns:
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                comp_idxs.append(r.completion_idx)
                turn_orders.append(info.get("turn_order", 0))
                envs.append(r.env)
                info_keys.append(info.get("info_key", ""))
                info_values.append(info.get("info_value", ""))
                info_types.append(info.get("info_type", "text"))
                tail_idxs.append(r.tail_idx)
        data = {
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "completion_idx": comp_idxs,
            "turn_order": turn_orders, "env": envs,
            "info_key": info_keys, "info_value": info_values,
            "info_type": info_types, "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=INFO_TURNS_EVAL_SCHEMA)

    def _sample_tags_eval_to_table(self, rollouts: list[EvalRollout]) -> pa.Table:
        """Convert eval rollouts' sample_tags to a normalized table."""
        if not rollouts:
            return pa.table({f.name: pa.array([], type=f.type) for f in SAMPLE_TAGS_EVAL_SCHEMA})
        steps, eval_names, sample_idxs, comp_idxs, envs, tag_names, tag_values, tail_idxs = [], [], [], [], [], [], [], []
        for r in rollouts:
            for tag_name, tag_value in r.sample_tags.items():
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                comp_idxs.append(r.completion_idx)
                envs.append(r.env)
                tag_names.append(tag_name)
                tag_values.append(str(tag_value))
                tail_idxs.append(r.tail_idx)
        data = {
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "completion_idx": comp_idxs,
            "env": envs, "tag_name": tag_names,
            "tag_value": tag_values, "tail_idx": tail_idxs,
        }
        return pa.table(data, schema=SAMPLE_TAGS_EVAL_SCHEMA)

    def _write_parquet_to_wandb(self, table: pa.Table, path: str):
        """Write a PyArrow table to W&B as a parquet file."""
        if self.run is None:
            return

        dest_path = Path(self.run.dir) / path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(table, str(dest_path))
        self.run.save(str(dest_path), base_path=self.run.dir, policy="now")

    def _compute_tail_idx_range(
        self,
        orchestrator_events: list[Event],
        trainer_events: list[Event],
        inference_events: list[InferenceEvent],
        gpu_metrics: list,
        cpu_metrics: list,
        vllm_metrics: list,
    ) -> tuple[int, int]:
        """
        Compute the min and max tail_idx from all data sources.
        
        Returns:
            (min_tail_idx, max_tail_idx) tuple, or (-1, -1) if no data
        """
        all_tail_indices = (
            [e.tail_idx for e in orchestrator_events] +
            [e.tail_idx for e in trainer_events] +
            [e.tail_idx for e in inference_events] +
            [idx for _, idx in gpu_metrics] +
            [idx for _, idx in cpu_metrics] +
            [idx for _, idx in vllm_metrics]
        )
        
        if all_tail_indices:
            return min(all_tail_indices), max(all_tail_indices)
        return -1, -1

    def _write_events_zip_to_wandb(
        self,
        orchestrator_events: list[Event],
        trainer_events: list[Event],
        inference_events: list[InferenceEvent],
        gpu_metrics: list,
        cpu_metrics: list,
        vllm_metrics: list,
        path: str,
        metadata: dict | None = None,
        discarded_rollouts: list[DiscardedRollout] | None = None,
        discarded_prompts: list[DiscardedPrompt] | None = None,
        eval_rollouts: list[EvalRollout] | None = None,
        eval_prompts: list[EvalPrompt] | None = None,
        logs: list[tuple[LogRecord, int]] | None = None,
        inflight_snapshot: list[dict] | None = None,
        inflight_compute_reward: list[dict] | None = None,
    ):
        """Write all event data as separate parquet files in a single zip archive."""
        if self.run is None:
            return

        dest_path = Path(self.run.dir) / path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert all data to tables
        orchestrator_table = self._orchestrator_events_to_table(orchestrator_events)
        trainer_table = self._trainer_events_to_table(trainer_events)
        inference_table = self._inference_events_to_table(inference_events)
        gpu_table = self._gpu_metrics_to_table(gpu_metrics)
        cpu_table = self._cpu_metrics_to_table(cpu_metrics)
        vllm_table = self._vllm_metrics_to_table(vllm_metrics)
        
        # Convert discarded prompts and rollouts (empty lists if None)
        discarded_proms = discarded_prompts or []
        discarded_gens = discarded_rollouts or []
        discarded_prompts_table = self._discarded_prompts_to_table(discarded_proms)
        discarded_gen_table = self._discarded_rollouts_to_table(discarded_gens)
        discarded_metrics_table = self._rollouts_metrics_discarded_to_table(discarded_gens)
        discarded_golden_answers_table = self._golden_answers_discarded_to_table(discarded_gens)
        discarded_samples_data_table = self._samples_data_discarded_to_table(discarded_gens)
        discarded_info_turns_table = self._info_turns_discarded_to_table(discarded_gens)
        discarded_sample_tags_table = self._sample_tags_discarded_to_table(discarded_gens)

        # Convert eval prompts and rollouts (empty lists if None)
        eval_rols = eval_rollouts or []
        eval_proms = eval_prompts or []
        eval_prompts_table = self._eval_prompts_to_table(eval_proms)
        eval_rollouts_table = self._eval_rollouts_to_table(eval_rols)
        eval_samples_data_table = self._samples_data_eval_to_table(eval_rols)
        eval_metrics_table = self._rollouts_metrics_eval_to_table(eval_rols)
        eval_golden_answers_table = self._golden_answers_eval_to_table(eval_rols)
        eval_info_turns_table = self._info_turns_eval_to_table(eval_rols)
        eval_sample_tags_table = self._sample_tags_eval_to_table(eval_rols)

        # Convert logs
        logs_list = logs or []
        logs_table = self._logs_to_table(logs_list)

        # Write all parquet files and metadata to a single zip
        with zipfile.ZipFile(dest_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            buf = io.BytesIO()
            pq.write_table(orchestrator_table, buf)
            zf.writestr("orchestrator.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(trainer_table, buf)
            zf.writestr("trainer.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(inference_table, buf)
            zf.writestr("inference.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(gpu_table, buf)
            zf.writestr("gpu.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cpu_table, buf)
            zf.writestr("cpu.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(vllm_table, buf)
            zf.writestr("vllm.parquet", buf.getvalue())

            # Discarded tables
            buf = io.BytesIO()
            pq.write_table(discarded_prompts_table, buf)
            zf.writestr("prompts_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_gen_table, buf)
            zf.writestr("rollouts_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_samples_data_table, buf)
            zf.writestr("samples_data_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_metrics_table, buf)
            zf.writestr("rollouts_metrics_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_golden_answers_table, buf)
            zf.writestr("golden_answers_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_info_turns_table, buf)
            zf.writestr("info_turns_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_sample_tags_table, buf)
            zf.writestr("sample_tags_discarded.parquet", buf.getvalue())

            # Eval tables
            buf = io.BytesIO()
            pq.write_table(eval_prompts_table, buf)
            zf.writestr("prompts_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_rollouts_table, buf)
            zf.writestr("rollouts_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_samples_data_table, buf)
            zf.writestr("samples_data_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_metrics_table, buf)
            zf.writestr("rollouts_metrics_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_golden_answers_table, buf)
            zf.writestr("golden_answers_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_info_turns_table, buf)
            zf.writestr("info_turns_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_sample_tags_table, buf)
            zf.writestr("sample_tags_eval.parquet", buf.getvalue())

            # Logs table
            buf = io.BytesIO()
            pq.write_table(logs_table, buf)
            zf.writestr("logs.parquet", buf.getvalue())

            # Write metadata as JSON file
            meta = dict(metadata) if metadata else {}
            meta["table_schema_versions"] = dict(table_schema_versions)
            zf.writestr("metadata.json", json.dumps(meta))

            # Write inflight snapshot if provided (only for tail.zip)
            if inflight_snapshot is not None:
                zf.writestr("inflight.json", json.dumps({
                    "snapshot_time": time.time(),
                    "running": inflight_snapshot,
                    "computing_reward": inflight_compute_reward or [],
                }))

        self.run.save(str(dest_path), base_path=self.run.dir, policy="now")

    def _write_steps_zip_to_wandb(
        self,
        rollouts: list[Rollout],
        prompts: list[Prompt],
        step_metrics: list[StepMetric],
        path: str,
        metadata: dict | None = None,
    ):
        """
        Write step data (prompts, rollouts, metrics) as parquet files inside a zip archive.
        
        Creates files in the zip:
        - prompts.parquet: One row per group with the initial prompt
        - rollouts.parquet: One row per turn per completion (model + env responses)
        - samples_data.parquet: One row per sample with reward, advantage, turn count, total_tokens, raw_string
        - rollouts_metrics.parquet: Normalized metrics table (step, sample_idx, env, metric_name, value)
        - golden_answers.parquet: Golden answers table (step, sample_idx, env, key, value)
        - info_turns.parquet: Per-turn text info table (step, sample_idx, turn_order, env, info_key, info_value, info_type)
        - sample_tags.parquet: Per-sample string tags table (step, sample_idx, env, tag_name, tag_value)
        - metrics.parquet: Per-step, per-rank metrics (step, metric, value, rank)
        - metadata.json: Archive metadata (if provided)
        """
        if self.run is None:
            return

        dest_path = Path(self.run.dir) / path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        prompts_table = self._prompts_to_table(prompts)
        gen_table = self._rollouts_to_table(rollouts)
        samples_data_table = self._samples_data_to_table(rollouts)
        gen_metrics_table = self._rollouts_metrics_to_table(rollouts)
        golden_answers_table = self._golden_answers_to_table(rollouts)
        info_turns_table = self._info_turns_to_table(rollouts)
        sample_tags_table = self._sample_tags_to_table(rollouts)
        metrics_table = self._step_metrics_to_table(step_metrics)

        with zipfile.ZipFile(dest_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Write prompts table (one row per group)
            buf = io.BytesIO()
            pq.write_table(prompts_table, buf)
            zf.writestr("prompts.parquet", buf.getvalue())
            
            # Write rollouts table (one row per turn)
            buf = io.BytesIO()
            pq.write_table(gen_table, buf)
            zf.writestr("rollouts.parquet", buf.getvalue())
            
            # Write samples_data table (one row per sample with summary info)
            buf = io.BytesIO()
            pq.write_table(samples_data_table, buf)
            zf.writestr("samples_data.parquet", buf.getvalue())
            
            # Write rollouts metrics table (normalized format: reward components + other metrics)
            buf = io.BytesIO()
            pq.write_table(gen_metrics_table, buf)
            zf.writestr("rollouts_metrics.parquet", buf.getvalue())
            
            # Write golden answers table (completely independent from metrics)
            buf = io.BytesIO()
            pq.write_table(golden_answers_table, buf)
            zf.writestr("golden_answers.parquet", buf.getvalue())
            
            # Write info turns table (per-turn text info: stderr, summaries, etc.)
            buf = io.BytesIO()
            pq.write_table(info_turns_table, buf)
            zf.writestr("info_turns.parquet", buf.getvalue())

            # Write sample tags table (per-sample string tags for filtering)
            buf = io.BytesIO()
            pq.write_table(sample_tags_table, buf)
            zf.writestr("sample_tags.parquet", buf.getvalue())

            # Write step metrics table (grad_norm, kl_divergence_inference, entropy per rank)
            buf = io.BytesIO()
            pq.write_table(metrics_table, buf)
            zf.writestr("metrics.parquet", buf.getvalue())

            # Write metadata as JSON file
            meta = dict(metadata) if metadata else {}
            meta["table_schema_versions"] = dict(table_schema_versions)
            zf.writestr("metadata.json", json.dumps(meta))

        self.run.save(str(dest_path), base_path=self.run.dir, policy="now")

    def _do_upload_sync(self):
        """
        Perform an upload cycle (blocking).
        
        This:
        1. Assigns tail_idx to all events/metrics that don't have one yet
        2. Gathers data from all sources (events, system metrics, vLLM metrics)
        3. Creates tail.zip with the last N complete tails (N = TAIL_WINDOW_SECONDS / UPLOAD_INTERVAL_SECONDS)
        4. Creates/updates block_live.zip with all complete tails in current block
        5. If block is complete (30 min), finalizes it as block_N.zip with complete tails
        6. Uploads any pending rollout files
        7. Updates run summary with metadata
        8. Increments tail_idx for the next upload cycle
        
        IMPORTANT: All filtering is done by tail_idx, not timestamp, to ensure complete tails.
        This prevents partial data at boundaries when the frontend fetches incrementally by tail_idx.
        """
        if self.run is None:
            return

        now = time.time()
        current_tail_idx = self._current_tail_idx

        # Assign tail_idx to events/inference_events/discarded_rollouts that don't have one yet
        with self._lock:
            for e in self._events:
                if e.tail_idx == -1:
                    e.tail_idx = current_tail_idx
            for e in self._inference_events:
                if e.tail_idx == -1:
                    e.tail_idx = current_tail_idx
            for g in self._discarded_rollouts:
                if g.tail_idx == -1:
                    g.tail_idx = current_tail_idx
            for p in self._discarded_prompts:
                if p.tail_idx == -1:
                    p.tail_idx = current_tail_idx
            for r in self._eval_rollouts:
                if r.tail_idx == -1:
                    r.tail_idx = current_tail_idx
            for p in self._eval_prompts:
                if p.tail_idx == -1:
                    p.tail_idx = current_tail_idx
            # Snapshot inflight generations for tail.zip
            inflight_snapshot = [
                {
                    "sample_id": e.sample_id,
                    "group_id": e.group_id,
                    "server": e.server,
                    "server_lane": e.server_lane,
                    "start_time": e.start_time,
                    "is_eval": e.is_eval,
                    "prompt_tokens": e.prompt_tokens,
                }
                for e in self._inflight_generations.values()
            ]
            # Snapshot inflight compute_reward for tail.zip
            inflight_compute_reward_snapshot = list(self._inflight_compute_reward.values())
            all_events = list(self._events)
            all_inference_events = list(self._inference_events)
            all_discarded_rollouts = list(self._discarded_rollouts)
            all_discarded_prompts = list(self._discarded_prompts)
            all_eval_rollouts = list(self._eval_rollouts)
            all_eval_prompts = list(self._eval_prompts)
            pending_gens = dict(self._pending_rollouts)
            self._pending_rollouts = {}
            pending_prompts = dict(self._pending_prompts)
            self._pending_prompts = {}
            pending_step_metrics = dict(self._pending_step_metrics)
            self._pending_step_metrics = {}
            trainer_steps_done = self._trainer_steps_done

        # Pull metrics from external loggers and assign tail_idx
        if self._system_metrics_logger is not None:
            new_gpu, new_cpu = self._system_metrics_logger.get_and_clear_metrics()
            # Assign current tail_idx to new metrics
            self._gpu_metrics.extend((m, current_tail_idx) for m in new_gpu)
            self._cpu_metrics.extend((m, current_tail_idx) for m in new_cpu)

        if self._vllm_metrics_logger is not None:
            new_vllm = self._vllm_metrics_logger.get_and_clear_metrics()
            self._vllm_metrics.extend((m, current_tail_idx) for m in new_vllm)

        # Pull log records from the logging system
        from telescope.utils import config as _config_mod
        cfg = getattr(_config_mod, 'cfg', None)
        upload_logs = getattr(cfg, 'wandb_upload_logs', False) if cfg else False
        upload_detailed = getattr(cfg, 'wandb_upload_logs_detailed', False) if cfg else False
        upload_stdout = getattr(cfg, 'wandb_upload_logs_stdout', False) if cfg else False

        if upload_logs or upload_detailed or upload_stdout:
            new_logs = drain_all_log_buffers()
            filtered_logs = []
            for rec in new_logs:
                if rec.source == "log" and upload_logs:
                    filtered_logs.append(rec)
                elif rec.source == "detailed" and upload_detailed:
                    filtered_logs.append(rec)
                elif rec.source == "stdout" and upload_stdout:
                    filtered_logs.append(rec)
            self._log_records.extend((rec, current_tail_idx) for rec in filtered_logs)

        # Assign tail_idx to any metrics that were added externally (via add_gpu_metrics)
        self._gpu_metrics = [(m, current_tail_idx if idx == -1 else idx) for m, idx in self._gpu_metrics]

        all_gpu_metrics = list(self._gpu_metrics)
        all_cpu_metrics = list(self._cpu_metrics)
        all_vllm_metrics = list(self._vllm_metrics)
        all_log_records = list(self._log_records)

        block_end_time = self._block_start_time + self.BLOCK_DURATION_SECONDS

        # Calculate tail_idx ranges for filtering (ensures complete tails, no partial data)
        # Tail window: last N tails where N = TAIL_WINDOW_SECONDS / UPLOAD_INTERVAL_SECONDS
        num_tails_in_window = self.TAIL_WINDOW_SECONDS // self.UPLOAD_INTERVAL_SECONDS
        tail_min_idx_for_window = max(0, current_tail_idx - num_tails_in_window + 1)

        # Check if we need to finalize the current block
        if now >= block_end_time:
            # Filter events for the completed block BY TAIL_IDX (ensures complete tails)
            # Block contains all tails from _block_first_tail_idx to current_tail_idx - 1
            # (current_tail_idx events are in the new block)
            block_last_tail_idx = current_tail_idx - 1
            
            block_events = [
                e for e in all_events
                if e.tail_idx >= self._block_first_tail_idx and e.tail_idx <= block_last_tail_idx
            ]
            block_inference_events = [
                e for e in all_inference_events
                if e.tail_idx >= self._block_first_tail_idx and e.tail_idx <= block_last_tail_idx
            ]
            block_gpu_metrics = [
                (m, idx) for m, idx in all_gpu_metrics
                if idx >= self._block_first_tail_idx and idx <= block_last_tail_idx
            ]
            block_cpu_metrics = [
                (m, idx) for m, idx in all_cpu_metrics
                if idx >= self._block_first_tail_idx and idx <= block_last_tail_idx
            ]
            block_vllm_metrics = [
                (m, idx) for m, idx in all_vllm_metrics
                if idx >= self._block_first_tail_idx and idx <= block_last_tail_idx
            ]
            block_log_records = [
                (r, idx) for r, idx in all_log_records
                if idx >= self._block_first_tail_idx and idx <= block_last_tail_idx
            ]
            block_discarded_rollouts = [
                g for g in all_discarded_rollouts
                if g.tail_idx >= self._block_first_tail_idx and g.tail_idx <= block_last_tail_idx
            ]
            block_discarded_prompts = [
                p for p in all_discarded_prompts
                if p.tail_idx >= self._block_first_tail_idx and p.tail_idx <= block_last_tail_idx
            ]
            block_eval_rollouts = [
                r for r in all_eval_rollouts
                if r.tail_idx >= self._block_first_tail_idx and r.tail_idx <= block_last_tail_idx
            ]
            block_eval_prompts = [
                p for p in all_eval_prompts
                if p.tail_idx >= self._block_first_tail_idx and p.tail_idx <= block_last_tail_idx
            ]

            if block_events or block_inference_events or block_gpu_metrics or block_cpu_metrics or block_vllm_metrics or block_discarded_rollouts or block_discarded_prompts or block_eval_rollouts or block_eval_prompts:
                orch_events = [e for e in block_events if e.source == "orchestrator"]
                trainer_events = [e for e in block_events if e.source == "trainer"]

                block_path = f"events/block_{self._current_block_idx}.zip"
                finalized_block_metadata = {
                    "block_idx": self._current_block_idx,
                    "min_tail_idx": self._block_first_tail_idx,
                    "max_tail_idx": block_last_tail_idx,
                }
                self._write_events_zip_to_wandb(
                    orch_events, trainer_events, block_inference_events,
                    block_gpu_metrics, block_cpu_metrics, block_vllm_metrics,
                    block_path,
                    metadata=finalized_block_metadata,
                    discarded_rollouts=block_discarded_rollouts,
                    discarded_prompts=block_discarded_prompts,
                    eval_rollouts=block_eval_rollouts,
                    eval_prompts=block_eval_prompts,
                    logs=block_log_records,
                )
                _log.debug(f"Finalized {block_path} (tails {self._block_first_tail_idx}-{block_last_tail_idx})")

            self._num_finalized_blocks = self._current_block_idx + 1
            self._current_block_idx += 1
            self._block_start_time = block_end_time
            
            # Update block_first_tail_idx for the new block
            new_block_first_tail_idx = current_tail_idx

            # Clean up old data BY TAIL_IDX (keeps only data for new block)
            # IMPORTANT: Re-assign tail_idx to any new events logged during this upload cycle
            # before cleanup, otherwise they'd be lost (they have tail_idx=-1)
            with self._lock:
                for e in self._events:
                    if e.tail_idx == -1:
                        e.tail_idx = current_tail_idx
                for e in self._inference_events:
                    if e.tail_idx == -1:
                        e.tail_idx = current_tail_idx
                for g in self._discarded_rollouts:
                    if g.tail_idx == -1:
                        g.tail_idx = current_tail_idx
                for p in self._discarded_prompts:
                    if p.tail_idx == -1:
                        p.tail_idx = current_tail_idx
                for r in self._eval_rollouts:
                    if r.tail_idx == -1:
                        r.tail_idx = current_tail_idx
                for p in self._eval_prompts:
                    if p.tail_idx == -1:
                        p.tail_idx = current_tail_idx
                self._events = [e for e in self._events if e.tail_idx >= new_block_first_tail_idx]
                self._inference_events = [e for e in self._inference_events if e.tail_idx >= new_block_first_tail_idx]
                self._discarded_rollouts = [g for g in self._discarded_rollouts if g.tail_idx >= new_block_first_tail_idx]
                self._discarded_prompts = [p for p in self._discarded_prompts if p.tail_idx >= new_block_first_tail_idx]
                self._eval_rollouts = [r for r in self._eval_rollouts if r.tail_idx >= new_block_first_tail_idx]
                self._eval_prompts = [p for p in self._eval_prompts if p.tail_idx >= new_block_first_tail_idx]
            
            # Also assign tail_idx to any new metrics added during this cycle
            self._gpu_metrics = [(m, current_tail_idx if idx == -1 else idx) for m, idx in self._gpu_metrics]
            self._gpu_metrics = [(m, idx) for m, idx in self._gpu_metrics if idx >= new_block_first_tail_idx]
            self._cpu_metrics = [(m, idx) for m, idx in self._cpu_metrics if idx >= new_block_first_tail_idx]
            self._vllm_metrics = [(m, idx) for m, idx in self._vllm_metrics if idx >= new_block_first_tail_idx]
            self._log_records = [(r, idx) for r, idx in self._log_records if idx >= new_block_first_tail_idx]

            all_events = [e for e in all_events if e.tail_idx >= new_block_first_tail_idx]
            all_inference_events = [e for e in all_inference_events if e.tail_idx >= new_block_first_tail_idx]
            all_gpu_metrics = [(m, idx) for m, idx in all_gpu_metrics if idx >= new_block_first_tail_idx]
            all_cpu_metrics = [(m, idx) for m, idx in all_cpu_metrics if idx >= new_block_first_tail_idx]
            all_vllm_metrics = [(m, idx) for m, idx in all_vllm_metrics if idx >= new_block_first_tail_idx]
            all_log_records = [(r, idx) for r, idx in all_log_records if idx >= new_block_first_tail_idx]
            all_discarded_rollouts = [g for g in all_discarded_rollouts if g.tail_idx >= new_block_first_tail_idx]
            all_eval_rollouts = [r for r in all_eval_rollouts if r.tail_idx >= new_block_first_tail_idx]
            all_eval_prompts = [p for p in all_eval_prompts if p.tail_idx >= new_block_first_tail_idx]
            
            # Set the new block's first tail_idx
            self._block_first_tail_idx = new_block_first_tail_idx

        # Filter for current block BY TAIL_IDX (ensures complete tails)
        current_block_events = [e for e in all_events if e.tail_idx >= self._block_first_tail_idx]
        current_block_inference_events = [e for e in all_inference_events if e.tail_idx >= self._block_first_tail_idx]
        current_block_gpu = [(m, idx) for m, idx in all_gpu_metrics if idx >= self._block_first_tail_idx]
        current_block_cpu = [(m, idx) for m, idx in all_cpu_metrics if idx >= self._block_first_tail_idx]
        current_block_vllm = [(m, idx) for m, idx in all_vllm_metrics if idx >= self._block_first_tail_idx]
        current_block_discarded_rollouts = [g for g in all_discarded_rollouts if g.tail_idx >= self._block_first_tail_idx]
        current_block_discarded_prompts = [p for p in all_discarded_prompts if p.tail_idx >= self._block_first_tail_idx]
        current_block_eval_rollouts = [r for r in all_eval_rollouts if r.tail_idx >= self._block_first_tail_idx]
        current_block_eval_prompts = [p for p in all_eval_prompts if p.tail_idx >= self._block_first_tail_idx]
        current_block_log_records = [(r, idx) for r, idx in all_log_records if idx >= self._block_first_tail_idx]

        # Filter for tail BY TAIL_IDX (ensures complete tails)
        tail_events = [e for e in all_events if e.tail_idx >= tail_min_idx_for_window]
        tail_inference_events = [e for e in all_inference_events if e.tail_idx >= tail_min_idx_for_window]
        tail_gpu = [(m, idx) for m, idx in all_gpu_metrics if idx >= tail_min_idx_for_window]
        tail_cpu = [(m, idx) for m, idx in all_cpu_metrics if idx >= tail_min_idx_for_window]
        tail_vllm = [(m, idx) for m, idx in all_vllm_metrics if idx >= tail_min_idx_for_window]
        tail_discarded_rollouts = [g for g in all_discarded_rollouts if g.tail_idx >= tail_min_idx_for_window]
        tail_discarded_prompts = [p for p in all_discarded_prompts if p.tail_idx >= tail_min_idx_for_window]
        tail_eval_rollouts = [r for r in all_eval_rollouts if r.tail_idx >= tail_min_idx_for_window]
        tail_eval_prompts = [p for p in all_eval_prompts if p.tail_idx >= tail_min_idx_for_window]
        tail_log_records = [(r, idx) for r, idx in all_log_records if idx >= tail_min_idx_for_window]

        # Split events by source
        tail_orch_events = [e for e in tail_events if e.source == "orchestrator"]
        tail_trainer_events = [e for e in tail_events if e.source == "trainer"]
        block_orch_events = [e for e in current_block_events if e.source == "orchestrator"]
        block_trainer_events = [e for e in current_block_events if e.source == "trainer"]

        # Calculate tail time range (from all data sources) - for informational purposes
        all_tail_timestamps = (
            [e.timestamp for e in tail_events] +
            [e.start_time for e in tail_inference_events] +
            [m.timestamp for m, _ in tail_gpu] +
            [m.timestamp for m, _ in tail_cpu] +
            [m.timestamp for m, _ in tail_vllm]
        )
        tail_start = min(all_tail_timestamps, default=now)
        tail_end = max(all_tail_timestamps, default=now)

        # Write tail.zip with all data
        # tail_idx range is from tail_min_idx_for_window to current_tail_idx (complete tails)
        tail_metadata = {
            "current_block_idx": self._current_block_idx,
            "tail_idx": current_tail_idx,
            "min_tail_idx": tail_min_idx_for_window,
            "max_tail_idx": current_tail_idx,
        }
        self._write_events_zip_to_wandb(
            tail_orch_events, tail_trainer_events, tail_inference_events,
            tail_gpu, tail_cpu, tail_vllm,
            "events/tail.zip",
            metadata=tail_metadata,
            discarded_rollouts=tail_discarded_rollouts,
            discarded_prompts=tail_discarded_prompts,
            eval_rollouts=tail_eval_rollouts,
            eval_prompts=tail_eval_prompts,
            logs=tail_log_records,
            inflight_snapshot=inflight_snapshot,
            inflight_compute_reward=inflight_compute_reward_snapshot,
        )

        # Write block_live.zip with all data
        # tail_idx range is from _block_first_tail_idx to current_tail_idx (complete tails)
        block_live_metadata = {
            "current_block_idx": self._current_block_idx,
            "tail_idx": current_tail_idx,
            "min_tail_idx": self._block_first_tail_idx,
            "max_tail_idx": current_tail_idx,
        }
        self._write_events_zip_to_wandb(
            block_orch_events, block_trainer_events, current_block_inference_events,
            current_block_gpu, current_block_cpu, current_block_vllm,
            "events/block_live.zip",
            metadata=block_live_metadata,
            discarded_rollouts=current_block_discarded_rollouts,
            discarded_prompts=current_block_discarded_prompts,
            eval_rollouts=current_block_eval_rollouts,
            eval_prompts=current_block_eval_prompts,
            logs=current_block_log_records,
        )

        # Process rollouts, prompts, and step metrics into blocks
        for step, gens in pending_gens.items():
            if gens:
                # Add to block data
                if step not in self._rollout_block_data:
                    self._rollout_block_data[step] = []
                self._rollout_block_data[step].extend(gens)
        
        for step, prompts in pending_prompts.items():
            if prompts:
                # Add to block data
                if step not in self._prompts_block_data:
                    self._prompts_block_data[step] = []
                self._prompts_block_data[step].extend(prompts)
        
        for step, metrics in pending_step_metrics.items():
            if metrics:
                # Add to block data
                if step not in self._step_metrics_block_data:
                    self._step_metrics_block_data[step] = []
                self._step_metrics_block_data[step].extend(metrics)

        # Check if we need to finalize a step block (every 500 steps)
        step_block_end_step = self._step_block_start_step + self.ROLLOUT_BLOCK_SIZE
        if self._last_training_step >= step_block_end_step:
            # Collect all rollouts, prompts, and metrics in the completed block
            block_rollouts = []
            block_prompts = []
            block_step_metrics = []
            steps_to_remove = []
            for step, gens in self._rollout_block_data.items():
                if step >= self._step_block_start_step and step < step_block_end_step:
                    block_rollouts.extend(gens)
                    steps_to_remove.append(step)
            for step, prompts in self._prompts_block_data.items():
                if step >= self._step_block_start_step and step < step_block_end_step:
                    block_prompts.extend(prompts)
            for step, metrics in self._step_metrics_block_data.items():
                if step >= self._step_block_start_step and step < step_block_end_step:
                    block_step_metrics.extend(metrics)

            if block_rollouts or block_prompts or block_step_metrics:
                block_path = f"steps/block_{self._current_step_block_idx}.zip"
                block_metadata = {
                    "block_idx": self._current_step_block_idx,
                    "start_step": self._step_block_start_step,
                    "end_step": step_block_end_step - 1,
                }
                self._write_steps_zip_to_wandb(block_rollouts, block_prompts, block_step_metrics, block_path, block_metadata)
                _log.debug(f"Finalized {block_path} with {len(block_prompts)} prompts, {len(block_rollouts)} samples, {len(block_step_metrics)} metrics (steps {self._step_block_start_step}-{step_block_end_step - 1})")

            # Remove finalized steps from block data
            for step in steps_to_remove:
                if step in self._rollout_block_data:
                    del self._rollout_block_data[step]
                if step in self._prompts_block_data:
                    del self._prompts_block_data[step]
                if step in self._step_metrics_block_data:
                    del self._step_metrics_block_data[step]

            # Advance to next block
            self._num_finalized_step_blocks = self._current_step_block_idx + 1
            self._current_step_block_idx += 1
            self._step_block_start_step = step_block_end_step

        # Write steps/tail.zip with the last 5 steps of data
        if self._last_training_step >= 0:
            # Collect from the last 5 steps
            all_steps = set(self._rollout_block_data.keys()) | set(self._step_metrics_block_data.keys()) | set(self._prompts_block_data.keys())
            available_steps = sorted(all_steps, reverse=True)[:5]
            tail_gens = []
            tail_prompts = []
            tail_step_metrics = []
            for step in available_steps:
                if step in self._rollout_block_data:
                    tail_gens.extend(self._rollout_block_data[step])
                if step in self._prompts_block_data:
                    tail_prompts.extend(self._prompts_block_data[step])
                if step in self._step_metrics_block_data:
                    tail_step_metrics.extend(self._step_metrics_block_data[step])
            if tail_gens or tail_prompts or tail_step_metrics:
                tail_gen_metadata = {
                    "min_step": min(available_steps),
                    "max_step": max(available_steps),
                }
                self._write_steps_zip_to_wandb(tail_gens, tail_prompts, tail_step_metrics, "steps/tail.zip", tail_gen_metadata)

        # Write steps/block_live.zip with current block's data
        current_block_rollouts = []
        current_block_prompts = []
        current_block_step_metrics = []
        current_block_steps = set()
        for step, gens in self._rollout_block_data.items():
            if step >= self._step_block_start_step:
                current_block_rollouts.extend(gens)
                current_block_steps.add(step)
        for step, prompts in self._prompts_block_data.items():
            if step >= self._step_block_start_step:
                current_block_prompts.extend(prompts)
                current_block_steps.add(step)
        for step, metrics in self._step_metrics_block_data.items():
            if step >= self._step_block_start_step:
                current_block_step_metrics.extend(metrics)
                current_block_steps.add(step)

        if current_block_rollouts or current_block_prompts or current_block_step_metrics:
            block_live_metadata = {
                "block_idx": self._current_step_block_idx,
                "min_step": min(current_block_steps) if current_block_steps else 0,
                "max_step": max(current_block_steps) if current_block_steps else 0,
            }
            self._write_steps_zip_to_wandb(current_block_rollouts, current_block_prompts, current_block_step_metrics, "steps/block_live.zip", block_live_metadata)

        # Get metadata from external loggers for summary
        num_gpus = 0
        num_vllm_servers = 0
        if self._system_metrics_logger is not None:
            num_gpus = self._system_metrics_logger.num_gpus
        if self._vllm_metrics_logger is not None:
            num_vllm_servers = self._vllm_metrics_logger.num_servers

        last_completed_training_step = trainer_steps_done - 1

        # Update run summary with consolidated metadata
        self.run.summary.update({
            "summary_id": self._summary_id,
            # Tail tracking
            "events/current_tail_idx": current_tail_idx,
            # Events block tracking
            "events/current_block_idx": self._current_block_idx,
            "events/num_finalized_blocks": self._num_finalized_blocks,
            "events/tail_start_time": tail_start,
            "events/tail_end_time": tail_end,
            # Events counts in tail
            "events/tail_orchestrator_count": len(tail_orch_events),
            "events/tail_trainer_count": len(tail_trainer_events),
            "events/tail_inference_count": len(tail_inference_events),
            "events/tail_gpu_count": len(tail_gpu),
            "events/tail_cpu_count": len(tail_cpu),
            "events/tail_vllm_count": len(tail_vllm),
            "events/tail_discarded_rollouts_count": len(tail_discarded_rollouts),
            # Events counts in block_live
            "events/block_live_orchestrator_count": len(block_orch_events),
            "events/block_live_trainer_count": len(block_trainer_events),
            "events/block_live_inference_count": len(current_block_inference_events),
            "events/block_live_gpu_count": len(current_block_gpu),
            "events/block_live_cpu_count": len(current_block_cpu),
            "events/block_live_vllm_count": len(current_block_vllm),
            "events/block_live_discarded_rollouts_count": len(current_block_discarded_rollouts),
            # Eval counts
            "events/tail_eval_rollouts_count": len(tail_eval_rollouts),
            "events/block_live_eval_rollouts_count": len(current_block_eval_rollouts),
            # System info
            "events/num_gpus": num_gpus,
            "events/num_vllm_servers": num_vllm_servers,
            "events/last_upload_time": now,
            # Steps tracking (rollouts + prompts + metrics)
            "steps/last_training_step": last_completed_training_step,
            "steps/current_block_idx": self._current_step_block_idx,
            "steps/num_finalized_blocks": self._num_finalized_step_blocks,
            "steps/block_start_step": self._step_block_start_step,
            "steps/block_live_prompt_count": len(current_block_prompts),
            "steps/block_live_rollout_count": len(current_block_rollouts),
            "steps/block_live_metrics_count": len(current_block_step_metrics),
        })

        self._last_upload_time = now
        # Increment summary_id for the next summary update
        self._summary_id += 1
        # Increment tail_idx for the next upload cycle
        self._current_tail_idx += 1
        _log.debug(f"Uploaded tail {current_tail_idx}: events={len(tail_orch_events)}+{len(tail_trainer_events)}+{len(tail_inference_events)}, metrics={len(tail_gpu)}+{len(tail_cpu)}+{len(tail_vllm)}, discarded={len(tail_discarded_rollouts)}")

    def upload_now(self, blocking: bool = False):
        """Trigger an upload cycle."""
        if blocking:
            self._do_upload_sync()
        else:
            if self._pending_upload is None or self._pending_upload.done():
                self._pending_upload = self._executor.submit(self._do_upload_sync)

    async def upload_now_async(self):
        """Async version that waits for upload to complete without blocking the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._do_upload_sync)

    async def start_upload_loop(self):
        """Start the background upload loop (runs every 5 seconds)."""
        self._stop_event = asyncio.Event()

        async def loop():
            next_upload_time = time.time()
            while not self._stop_event.is_set():
                # Schedule next upload at a fixed interval from the start
                next_upload_time += self.UPLOAD_INTERVAL_SECONDS
                
                try:
                    await self.upload_now_async()
                except Exception as e:
                    _log.error(f"Upload error: {e}")

                # Wait until the next scheduled upload time
                wait_time = max(0, next_upload_time - time.time())
                if wait_time > 0:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=wait_time
                        )
                    except asyncio.TimeoutError:
                        pass

        self._upload_task = asyncio.create_task(loop())
        _log.debug("Event upload loop started")

    async def stop_upload_loop(self):
        """Stop the background upload loop."""
        if self._stop_event:
            self._stop_event.set()
        if self._upload_task:
            await self._upload_task
            self._upload_task = None
        _log.info("Event upload loop stopped")

    def finish(self):
        """Finalize and upload any remaining data."""
        if self._pending_upload is not None:
            try:
                self._pending_upload.result(timeout=30)
            except Exception as e:
                _log.error(f"Pending upload error: {e}")

        self.upload_now(blocking=True)
        self._executor.shutdown(wait=True)
        _log.info("EventLogger finished")

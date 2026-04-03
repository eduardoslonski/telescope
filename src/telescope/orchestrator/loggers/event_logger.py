"""
Event logging system for training UI visualization.

This module implements a parquet-based event logging system that uploads to W&B
in a way that's easy for the frontend UI to fetch and filter.

All events are stored together in unified zip archives:
- events/tail.zip: Last 60 seconds of all data
  Contains: orchestrator.parquet, trainer.parquet, events_rollout.parquet, events_infra.parquet, gpu.parquet, cpu.parquet, vllm.parquet, thread_pools.parquet,
            generations_discarded.parquet, env_responses_discarded.parquet, tool_calls_discarded.parquet,
            rollouts_metrics_discarded.parquet, golden_answers_discarded.parquet, info_turns_discarded.parquet,
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

Rollout events track per-sample rollout pipeline phases (generation, tool_execution, env_response, reward):
- event_type, phase, sample_id, generation_idx, tool_call_idx, server_id, agent_id
- tail_idx: Upload cycle index when first uploaded

Infra events track infrastructure operations (weight_sync, sandbox_provision):
- event_type, phase, step, server_id, sandbox_id
- tail_idx: Upload cycle index when first uploaded

Steps data is stored in blocks by step count:
- steps/tail.zip: Last 5 training steps' data (rollouts, rewards, metrics)
- steps/block_live.zip: Current block being built (up to 500 steps)
- steps/block_*.zip: Finalized blocks (every 500 steps)

Run summary is updated every 5 seconds with metadata to help the UI:
- events/current_tail_idx: Current tail upload cycle index
- events/current_block_idx: Which event block we're currently writing to
- events/num_finalized_blocks: How many event block_*.zip files have been written
- events/tail_rollout_events_count: Number of rollout events in tail
- events/tail_infra_events_count: Number of infra events in tail
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
from telescope.orchestrator.loggers.thread_pool_metrics_logger import (
    ThreadPoolMetricsLogger,
    ThreadPoolMetricSample,
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

# Rollout events schema - per-sample lifecycle events (generation, tool execution, env response, reward)
# Start and end are separate rows. End rows also carry start_time for self-contained historical queries.
EVENTS_ROLLOUT_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),   # Event timestamp (start_time for start rows, end_time for end rows)
    ("tail_idx", pa.int32()),
    ("event_type", pa.string()),   # "generation", "tool_execution", "env_response", "reward"
    ("phase", pa.string()),        # "start" or "end"
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),     # Run-wide unique sample ID
    ("agent_id", pa.int32()),      # Agent ID (0 = main agent, 1+ = sub-agents)
    ("generation_idx", pa.int32()),  # Generation index within (sample_id, agent_id)
    ("tool_call_idx", pa.int32()),   # Tool call index within generation (-1 if N/A)
    ("server_id", pa.int32()),       # Inference server index (-1 if N/A)
    ("server_lane", pa.int32()),     # Per-server lane slot for timeline positioning (-1 if N/A)
])

# Infrastructure events schema - weight sync, sandbox lifecycle
EVENTS_INFRA_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("tail_idx", pa.int32()),
    ("event_type", pa.string()),   # "weight_sync", "sandbox"
    ("phase", pa.string()),        # "start"/"end" for weight_sync; "create"/"setup"/"ready"/"execute"/"destroy" for sandbox
    ("step", pa.int32()),          # Training step (-1 if N/A)
    ("server_id", pa.int32()),     # For weight_sync events (-1 if N/A)
    ("sandbox_id", pa.string()),   # For sandbox events (null if N/A)
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

# Thread pool metrics schema
THREAD_POOL_METRICS_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("pool_name", pa.string()),
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

# Generations schema - one row per model generation
# For single-turn: one row per sample with generation_idx=0
# For multi-turn: multiple rows per sample, one per model generation
GENERATIONS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),      # Links to prompts table
    ("sample_id", pa.int32()),     # Run-wide unique sample ID
    ("agent_id", pa.int32()),      # 0 = main agent, 1+ = sub-agents
    ("generation_idx", pa.int32()),  # 0-indexed within (sample_id, agent_id)
    ("content", pa.binary()),      # Full generation text (zstd-compressed UTF-8)
    ("tokens", pa.int32()),        # Tokens generated
    ("prompt_tokens", pa.int32()), # Tokens in the prompt for this generation
    ("tool_call_count", pa.int32()),  # Number of tool calls parsed (0 for reasoning)
    ("stop_reason", pa.string()),  # "eos", "max_tokens", "tool_call", "tool_result_interrupt"
    # vLLM per-request timing
    ("queue_time", pa.float64()),
    ("ttft", pa.float64()),        # Time to first token
    ("prefill_time", pa.float64()),
    ("decode_time", pa.float64()),
    ("inference_time", pa.float64()),  # prefill + decode
    ("e2e_latency", pa.float64()),
    ("server_id", pa.int32()),
    ("vllm_request_id", pa.string()),
])

# Environment responses schema - one row per injection between generations
ENV_RESPONSES_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),  # Which generation this follows
    ("content", pa.binary()),        # Text injected into conversation (zstd-compressed UTF-8)
    ("turn_type", pa.string()),      # "tool_result", "env_response", "feedback", "context"
    ("tokens", pa.int32()),
    ("response_time", pa.float64()), # Wall clock for entire env_response processing
])

# Tool calls schema - one row per tool call (call + result combined)
TOOL_CALLS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),              # Which generation created this call
    ("tool_call_idx", pa.int32()),               # 0-indexed within generation
    ("env_response_generation_idx", pa.int32()), # Which env_response contains the result
    ("tool_name", pa.string()),
    ("arguments", pa.string()),                  # JSON
    ("raw_text", pa.string()),                   # Original parsed text from generation
    ("result", pa.binary()),                     # Tool output (zstd-compressed, can be large)
    ("success", pa.bool_()),
    ("error", pa.string()),
    ("exit_code", pa.int32()),                   # For bash/terminal tools (-1 if N/A)
    ("truncated", pa.bool_()),
    ("result_tokens", pa.int32()),
    ("sandbox_id", pa.string()),
])

# Sandboxes schema - one row per sandbox session
SANDBOXES_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),
    ("sandbox_id", pa.string()),
    ("sandbox_type", pa.string()),  # "docker", "modal", "firecracker", "e2b", "namespace"
    ("image", pa.string()),
    ("create_time", pa.float64()),
    ("setup_time", pa.float64()),
    ("total_execute_time", pa.float64()),
    ("destroy_time", pa.float64()),
    ("cpu_limit", pa.float32()),
    ("memory_limit_mb", pa.int32()),
    ("disk_limit_mb", pa.int32()),
    ("timeout_seconds", pa.int32()),
    ("peak_cpu_pct", pa.float32()),
    ("peak_memory_mb", pa.int32()),
    ("peak_disk_mb", pa.int32()),
    ("status", pa.string()),       # "created", "ready", "executing", "completed", "failed", "timeout"
    ("error", pa.string()),
    ("tool_calls_executed", pa.int32()),
    ("reused", pa.bool_()),
])

# Turn metrics schema - per-generation or per-env-response metrics
TURN_METRICS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("turn_type", pa.string()),    # "generation" or "env_response"
    ("env", pa.string()),
    ("metric_name", pa.string()),
    ("value", pa.float64()),
])

# Samples data schema - one row per sample with summary information
# Links to prompts table via group_id, to generations/env_responses/tool_calls via sample_id
SAMPLES_DATA_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("group_id", pa.int32()),      # Links to prompts table
    ("sample_id", pa.int32()),     # Run-wide unique sample ID
    ("reward", pa.float64()),      # Total reward
    ("advantage", pa.float64()),   # Computed advantage
    ("num_generations", pa.int32()),  # Number of model generations in this sample
    ("total_tokens", pa.int32()),  # Total tokens passed to trainer
    ("raw_string", pa.binary()),   # Decoded raw input passed to trainer (zstd-compressed UTF-8)
    ("compute_reward_time", pa.float64()),  # Time in seconds for compute_reward() call
    ("stop_reason", pa.string()),  # Why rollout ended: "final_answer", "max_turns", "context_exhausted", "error"
    ("off_policy_steps", pa.int32()),  # Number of weight updates since this rollout was dispatched
])

# Rollouts metrics schema - normalized table for flexible per-sample metrics
# Includes both reward components and other sample metrics
ROLLOUTS_METRICS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_id", pa.int32()),     # Run-wide unique sample ID
    ("env", pa.string()),
    ("metric_name", pa.string()),
    ("value", pa.float64()),
])

# Golden answers schema - ground truth answers per sample
GOLDEN_ANSWERS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_id", pa.int32()),
    ("env", pa.string()),
    ("key", pa.string()),
    ("value", pa.string()),
])

# Sample tags schema - per-sample string tags for filtering
SAMPLE_TAGS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_id", pa.int32()),
    ("env", pa.string()),
    ("tag_name", pa.string()),
    ("tag_value", pa.string()),
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

# Discarded generations schema
GENERATIONS_DISCARDED_SCHEMA = pa.schema([
    ("trainer_step", pa.int32()),
    ("inference_step", pa.int32()),
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("content", pa.binary()),
    ("tokens", pa.int32()),
    ("prompt_tokens", pa.int32()),
    ("tool_call_count", pa.int32()),
    ("stop_reason", pa.string()),
    ("queue_time", pa.float64()),
    ("ttft", pa.float64()),
    ("prefill_time", pa.float64()),
    ("decode_time", pa.float64()),
    ("inference_time", pa.float64()),
    ("e2e_latency", pa.float64()),
    ("server_id", pa.int32()),
    ("vllm_request_id", pa.string()),
    ("tail_idx", pa.int32()),
])

# Discarded environment responses schema
ENV_RESPONSES_DISCARDED_SCHEMA = pa.schema([
    ("trainer_step", pa.int32()),
    ("inference_step", pa.int32()),
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("content", pa.binary()),
    ("turn_type", pa.string()),
    ("tokens", pa.int32()),
    ("response_time", pa.float64()),
    ("tail_idx", pa.int32()),
])

# Discarded tool calls schema
TOOL_CALLS_DISCARDED_SCHEMA = pa.schema([
    ("trainer_step", pa.int32()),
    ("inference_step", pa.int32()),
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("tool_call_idx", pa.int32()),
    ("env_response_generation_idx", pa.int32()),
    ("tool_name", pa.string()),
    ("arguments", pa.string()),
    ("raw_text", pa.string()),
    ("result", pa.binary()),
    ("success", pa.bool_()),
    ("error", pa.string()),
    ("exit_code", pa.int32()),
    ("truncated", pa.bool_()),
    ("result_tokens", pa.int32()),
    ("sandbox_id", pa.string()),
    ("tail_idx", pa.int32()),
])

# Discarded samples data schema
SAMPLES_DATA_DISCARDED_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("discard_reason", pa.string()),
    ("trainer_step", pa.int32()),
    ("inference_step", pa.int32()),
    ("group_id", pa.int32()),
    ("sample_id", pa.int32()),
    ("reward", pa.float64()),
    ("advantage", pa.float64()),
    ("num_generations", pa.int32()),
    ("total_tokens", pa.int32()),
    ("raw_string", pa.binary()),
    ("compute_reward_time", pa.float64()),
    ("stop_reason", pa.string()),
    ("off_policy_steps", pa.int32()),
    ("tail_idx", pa.int32()),
])

# Discarded rollouts metrics schema
ROLLOUTS_METRICS_DISCARDED_SCHEMA = pa.schema([
    ("sample_id", pa.int32()),
    ("env", pa.string()),
    ("metric_name", pa.string()),
    ("value", pa.float64()),
    ("tail_idx", pa.int32()),
])

# Discarded golden answers schema
GOLDEN_ANSWERS_DISCARDED_SCHEMA = pa.schema([
    ("sample_id", pa.int32()),
    ("env", pa.string()),
    ("key", pa.string()),
    ("value", pa.string()),
    ("tail_idx", pa.int32()),
])

# Info turns schema - per-generation text information for a sample
# One row per info item. Multiple info items can exist per generation.
# Example uses: stderr from code execution, model summaries, debug output, etc.
INFO_TURNS_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),   # Which generation this info is for
    ("tool_call_idx", pa.int32()),    # Which tool call (-1 if not tool-specific)
    ("env", pa.string()),
    ("info_key", pa.string()),
    ("info_value", pa.string()),
    ("info_type", pa.string()),
])

# Discarded info turns schema
INFO_TURNS_DISCARDED_SCHEMA = pa.schema([
    ("sample_id", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("tool_call_idx", pa.int32()),
    ("env", pa.string()),
    ("info_key", pa.string()),
    ("info_value", pa.string()),
    ("info_type", pa.string()),
    ("tail_idx", pa.int32()),
])

# Discarded sample tags schema
SAMPLE_TAGS_DISCARDED_SCHEMA = pa.schema([
    ("sample_id", pa.int32()),
    ("env", pa.string()),
    ("tag_name", pa.string()),
    ("tag_value", pa.string()),
    ("tail_idx", pa.int32()),
])

# ---- Eval schemas (parallel to discarded, for eval completions) ----

PROMPTS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("model_step", pa.int32()),
    ("sample_idx", pa.int32()),   # Index in the eval dataset (not renamed, eval-specific)
    ("sample_id", pa.int32()),    # Globally unique sample ID (correlates with events_rollout)
    ("env", pa.string()),
    ("prompt", pa.string()),
    ("tokens_prompt", pa.int32()),
    ("system_prompt", pa.string()),
    ("tokens_system_prompt", pa.int32()),
    ("tail_idx", pa.int32()),
])

GENERATIONS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("model_step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("sample_id", pa.int32()),
    ("completion_idx", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("content", pa.binary()),
    ("tokens", pa.int32()),
    ("prompt_tokens", pa.int32()),
    ("tool_call_count", pa.int32()),
    ("stop_reason", pa.string()),
    ("tail_idx", pa.int32()),
])

ENV_RESPONSES_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("model_step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("sample_id", pa.int32()),
    ("completion_idx", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("content", pa.binary()),
    ("turn_type", pa.string()),
    ("tokens", pa.int32()),
    ("response_time", pa.float64()),
    ("tail_idx", pa.int32()),
])

TOOL_CALLS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("model_step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("sample_id", pa.int32()),
    ("completion_idx", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("tool_call_idx", pa.int32()),
    ("env_response_generation_idx", pa.int32()),
    ("tool_name", pa.string()),
    ("arguments", pa.string()),
    ("raw_text", pa.string()),
    ("result", pa.binary()),
    ("success", pa.bool_()),
    ("error", pa.string()),
    ("exit_code", pa.int32()),
    ("truncated", pa.bool_()),
    ("result_tokens", pa.int32()),
    ("sandbox_id", pa.string()),
    ("tail_idx", pa.int32()),
])

SAMPLES_DATA_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("model_step", pa.int32()),
    ("sample_idx", pa.int32()),
    ("sample_id", pa.int32()),
    ("completion_idx", pa.int32()),
    ("env", pa.string()),
    ("num_generations", pa.int32()),
    ("compute_eval_metrics_time", pa.float64()),
    ("stop_reason", pa.string()),
    ("tail_idx", pa.int32()),
])

ROLLOUTS_METRICS_EVAL_SCHEMA = pa.schema([
    ("step", pa.int32()),
    ("eval_name", pa.string()),
    ("sample_idx", pa.int32()),
    ("sample_id", pa.int32()),
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
    ("sample_id", pa.int32()),
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
    ("sample_id", pa.int32()),
    ("completion_idx", pa.int32()),
    ("agent_id", pa.int32()),
    ("generation_idx", pa.int32()),
    ("tool_call_idx", pa.int32()),
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
    ("sample_id", pa.int32()),
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
class GenerationRecord:
    """One model generation (one vLLM request/response)."""
    generation_idx: int  # 0-indexed within (sample_id, agent_id)
    content: str  # Full generation text
    tokens: int = 0  # Tokens generated
    prompt_tokens: int = 0  # Tokens in prompt for this generation
    tool_call_count: int = 0  # Number of tool calls parsed
    stop_reason: str = ""  # "eos", "max_tokens", "tool_call", "tool_result_interrupt"
    # vLLM timing
    queue_time: float = 0.0
    ttft: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    inference_time: float = 0.0
    e2e_latency: float = 0.0
    server_id: int = -1
    vllm_request_id: str = ""


@dataclass
class EnvResponseRecord:
    """One environment response between generations."""
    generation_idx: int  # Which generation this follows
    content: str  # Text injected into conversation
    turn_type: str = "env_response"  # "tool_result", "env_response", "feedback", "context"
    tokens: int = 0
    response_time: float = 0.0  # Wall clock for entire env_response processing


@dataclass
class ToolCallRecord:
    """One tool call with its result."""
    generation_idx: int  # Which generation created this call
    tool_call_idx: int  # 0-indexed within generation
    env_response_generation_idx: int  # Which env_response contains the result
    tool_name: str = ""
    arguments: str = ""  # JSON
    raw_text: str = ""  # Original parsed text from generation
    result: str = ""  # Tool output text
    success: bool = True
    error: str = ""
    exit_code: int = -1  # For bash/terminal tools
    truncated: bool = False
    result_tokens: int = 0
    sandbox_id: str = ""


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
    tail_idx: int = -1  # Index of the tail upload cycle when first uploaded (events system)


@dataclass
class GenerationRollout:
    """Complete rollout data for one sample, structured around generations."""
    step: int
    group_id: int
    sample_id: int  # Run-wide unique sample ID
    agent_id: int = 0
    env: str = ""
    generations: list[GenerationRecord] = field(default_factory=list)
    env_responses: list[EnvResponseRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    reward: float = 0.0
    advantage: float = 0.0
    sample_metrics: dict[str, float] = field(default_factory=dict)
    golden_answers: dict[str, str | None] = field(default_factory=dict)
    info_turns: list[dict] = field(default_factory=list)
    sample_tags: dict[str, str] = field(default_factory=dict)
    turn_metrics: list[dict] = field(default_factory=list)  # Per-generation/env-response metrics
    total_tokens: int = 0
    raw_string: str = ""
    compute_reward_time: float = 0.0
    stop_reason: str = ""  # Why rollout ended: "final_answer", "max_turns", etc.
    off_policy_steps: int = 0  # Number of weight updates since this rollout was dispatched
    tail_idx: int = -1


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
class DiscardedGenerationRollout:
    """A rollout sample that was discarded (not sent to trainer)."""
    timestamp: float
    discard_reason: str
    trainer_step: int
    inference_step: int
    group_id: int
    sample_id: int
    agent_id: int = 0
    env: str = ""
    generations: list[GenerationRecord] = field(default_factory=list)
    env_responses: list[EnvResponseRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    reward: float = 0.0
    advantage: float = 0.0
    sample_metrics: dict[str, float] = field(default_factory=dict)
    golden_answers: dict[str, str | None] = field(default_factory=dict)
    info_turns: list[dict] = field(default_factory=list)
    sample_tags: dict[str, str] = field(default_factory=dict)
    total_tokens: int = 0
    raw_string: str = ""
    compute_reward_time: float = 0.0
    stop_reason: str = ""
    off_policy_steps: int = 0
    tail_idx: int = -1


@dataclass
class RolloutEvent:
    """Per-sample lifecycle event (generation, tool_execution, env_response, reward)."""
    timestamp: float
    event_type: str  # "generation", "tool_execution", "env_response", "reward"
    phase: str  # "start" or "end"
    group_id: int = -1
    sample_id: int = -1
    agent_id: int = 0
    generation_idx: int = -1
    tool_call_idx: int = -1
    server_id: int = -1
    server_lane: int = -1  # Per-server lane slot for timeline positioning
    tail_idx: int = -1


@dataclass
class InfraEvent:
    """Infrastructure lifecycle event (weight_sync, sandbox)."""
    timestamp: float
    event_type: str  # "weight_sync", "sandbox"
    phase: str  # "start"/"end" for weight_sync; "create"/"setup"/"ready"/"execute"/"destroy" for sandbox
    step: int = -1
    server_id: int = -1
    sandbox_id: str = ""
    tail_idx: int = -1


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
    sample_id: int = -1  # Globally unique sample ID (correlates with events_rollout)
    tail_idx: int = -1


@dataclass
class EvalGenerationRollout:
    """A single eval completion structured around generations."""
    step: int
    eval_name: str
    model_step: int
    sample_idx: int
    completion_idx: int  # 0..max(pass_k)-1
    env: str = ""
    agent_id: int = 0
    sample_id: int = -1  # Globally unique sample ID (correlates with events_rollout)
    generations: list[GenerationRecord] = field(default_factory=list)
    env_responses: list[EnvResponseRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sample_metrics: dict[str, float] = field(default_factory=dict)
    golden_answers: dict[str, str | None] = field(default_factory=dict)
    info_turns: list[dict] = field(default_factory=list)
    sample_tags: dict[str, str] = field(default_factory=dict)
    compute_eval_metrics_time: float = 0.0
    stop_reason: str = ""
    tail_idx: int = -1


class EventLogger:
    """
    Thread-safe event logger that buffers events and periodically uploads to W&B.
    
    This logger consolidates data from multiple sources:
    - Orchestrator events (instant events)
    - Trainer events (duration events with hierarchy)
    - Rollout events (generation/tool_execution/env_response/reward lifecycle)
    - Infrastructure events (weight_sync, sandbox lifecycle)
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
        logger.log_rollout_event(event_type="generation", phase="start", ...)

        # Log rollouts
        logger.log_generation_rollout(step=0, sample_id=0, generations=[...], ...)

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
        self._rollout_events: list[RolloutEvent] = []
        self._infra_events: list[InfraEvent] = []

        # Inflight tracking (for snapshot in tail.zip)
        self._inflight_generations: dict[int, dict] = {}  # sample_id -> {sample_id, generation_idx, server_id, agent_id}
        self._inflight_tool_executions: dict[tuple[int, int], dict] = {}  # (sample_id, tool_call_idx) -> {sample_id, generation_idx, tool_call_idx, tool_name, agent_id}
        self._inflight_env_responses: dict[int, dict] = {}  # sample_id -> {sample_id, generation_idx, agent_id}
        self._inflight_rewards: dict[int, dict] = {}  # sample_id -> {sample_id, agent_id}
        self._inflight_weight_syncs: dict[int, dict] = {}  # server_id -> {server_id, step}
        self._inflight_sandbox_ops: dict[str, dict] = {}  # sandbox_id -> {sandbox_id, phase}

        # External metrics loggers (set via set_metrics_loggers)
        self._system_metrics_logger: SystemMetricsLogger | None = None
        self._vllm_metrics_logger: VllmMetricsLogger | None = None
        self._thread_pool_metrics_logger: ThreadPoolMetricsLogger | None = None

        # Accumulated metrics from external loggers (persisted across uploads for block management)
        # Each metric is stored as (metric, tail_idx) tuple
        self._gpu_metrics: list[tuple[GpuMetricSample, int]] = []
        self._cpu_metrics: list[tuple[CpuMetricSample, int]] = []
        self._vllm_metrics: list[tuple[VllmMetricSample, int]] = []
        self._thread_pool_metrics: list[tuple[ThreadPoolMetricSample, int]] = []

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
        self._pending_rollouts: dict[int, list[GenerationRollout]] = {}
        self._pending_prompts: dict[int, list[Prompt]] = {}  # Prompts per step
        self._logged_group_ids: set[int] = set()  # Track which group_ids have been logged
        
        # Step metrics tracking (grad_norm, kl_divergence_inference, entropy per rank)
        self._pending_step_metrics: dict[int, list[StepMetric]] = {}
        
        # Step block tracking (by step count, not time) - includes rollouts, prompts, and metrics
        self._rollout_block_data: dict[int, list[GenerationRollout]] = {}  # All rollouts in current block
        self._prompts_block_data: dict[int, list[Prompt]] = {}  # All prompts in current block
        self._step_metrics_block_data: dict[int, list[StepMetric]] = {}  # All step metrics in current block
        self._step_block_start_step: int = 0  # First step in current block
        self._current_step_block_idx: int = 0
        self._num_finalized_step_blocks: int = 0

        # Discarded rollouts tracking (uploaded in events zip, not rollouts)
        # These are rollouts that were not sent to the trainer (async limit, zero advantage, etc.)
        self._discarded_rollouts: list[DiscardedGenerationRollout] = []
        self._discarded_prompts: list[DiscardedPrompt] = []
        self._logged_discarded_group_ids: set[int] = set()  # Track which discarded group_ids have been logged

        # Cancelled rollouts tracking (uploaded in events zip, same schema as discarded)
        # These are rollouts that were interrupted mid-flight (off-policy cancellation, eval drain, etc.)
        self._cancelled_rollouts: list[DiscardedGenerationRollout] = []
        self._cancelled_prompts: list[DiscardedPrompt] = []
        self._logged_cancelled_group_ids: set[int] = set()

        # Kept rollouts tracking (uploaded in events zip alongside discarded, for real-time incremental ingestion)
        self._kept_rollouts: list[GenerationRollout] = []
        self._kept_prompts: list[Prompt] = []

        # Eval tracking (parallel to discarded, in events zip)
        self._eval_rollouts: list[EvalGenerationRollout] = []
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
        thread_pool_metrics_logger: ThreadPoolMetricsLogger | None = None,
    ):
        """
        Set references to external metrics loggers.

        EventLogger will pull metrics from these loggers during upload cycles.

        Args:
            system_metrics_logger: SystemMetricsLogger instance for GPU/CPU metrics
            vllm_metrics_logger: VllmMetricsLogger instance for vLLM metrics
            thread_pool_metrics_logger: ThreadPoolMetricsLogger instance for thread pool metrics
        """
        self._system_metrics_logger = system_metrics_logger
        self._vllm_metrics_logger = vllm_metrics_logger
        self._thread_pool_metrics_logger = thread_pool_metrics_logger
        _log.debug(
            f"Set metrics loggers: system={system_metrics_logger is not None}, "
            f"vllm={vllm_metrics_logger is not None}, "
            f"thread_pools={thread_pool_metrics_logger is not None}"
        )

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

    def log_generation_rollout(
        self,
        step: int,
        group_id: int,
        sample_id: int,
        prompt: str,
        generations: list[dict] | None = None,
        env_responses: list[dict] | None = None,
        tool_calls: list[dict] | None = None,
        reward: float = 0.0,
        advantage: float = 0.0,
        env: str = "",
        sample_metrics: dict[str, float] | None = None,
        golden_answers: dict[str, str | None] | None = None,
        info_turns: list[dict] | None = None,
        sample_tags: dict[str, str] | None = None,
        turn_metrics: list[dict] | None = None,
        tokens_prompt: int = 0,
        system_prompt: str = "",
        tokens_system_prompt: int = 0,
        total_tokens: int = 0,
        raw_string: str = "",
        compute_reward_time: float = 0.0,
        stop_reason: str = "",
        off_policy_steps: int = 0,
        agent_id: int = 0,
    ):
        """
        Log a generation-centric rollout sample for a training step.

        Args:
            step: Training step
            group_id: Request group ID
            sample_id: Run-wide unique sample ID
            prompt: Input prompt text
            generations: List of generation dicts (generation_idx, content, tokens, prompt_tokens, ...)
            env_responses: List of env response dicts (generation_idx, content, turn_type, tokens, ...)
            tool_calls: List of tool call dicts (generation_idx, tool_call_idx, tool_name, ...)
            reward: Total reward
            advantage: Computed advantage
            env: Environment name
            sample_metrics: Per-sample metrics
            golden_answers: Golden answers
            info_turns: Per-generation info items
            sample_tags: Per-sample tags
            turn_metrics: Per-generation/env-response metrics
            tokens_prompt: Number of initial prompt tokens
            system_prompt: System message
            tokens_system_prompt: System message tokens
            total_tokens: Total tokens passed to trainer
            raw_string: Raw input passed to trainer
            compute_reward_time: Reward computation time
            stop_reason: Why the rollout ended
            agent_id: Agent ID (0 = main)
        """
        gen_records = [
            GenerationRecord(
                generation_idx=g["generation_idx"],
                content=g.get("content", ""),
                tokens=g.get("tokens", 0),
                prompt_tokens=g.get("prompt_tokens", 0),
                tool_call_count=g.get("tool_call_count", 0),
                stop_reason=g.get("stop_reason", ""),
                queue_time=g.get("queue_time", 0.0),
                ttft=g.get("ttft", 0.0),
                prefill_time=g.get("prefill_time", 0.0),
                decode_time=g.get("decode_time", 0.0),
                inference_time=g.get("inference_time", 0.0),
                e2e_latency=g.get("e2e_latency", 0.0),
                server_id=g.get("server_id", -1),
                vllm_request_id=g.get("vllm_request_id", ""),
            )
            for g in (generations or [])
        ]

        env_records = [
            EnvResponseRecord(
                generation_idx=e["generation_idx"],
                content=e.get("content", ""),
                turn_type=e.get("turn_type", "env_response"),
                tokens=e.get("tokens", 0),
                response_time=e.get("response_time", 0.0),
            )
            for e in (env_responses or [])
        ]

        tc_records = [
            ToolCallRecord(
                generation_idx=tc["generation_idx"],
                tool_call_idx=tc.get("tool_call_idx", 0),
                env_response_generation_idx=tc.get("env_response_generation_idx", tc["generation_idx"]),
                tool_name=tc.get("tool_name", ""),
                arguments=tc.get("arguments", ""),
                raw_text=tc.get("raw_text", ""),
                result=tc.get("result", ""),
                success=tc.get("success", True),
                error=tc.get("error", ""),
                exit_code=tc.get("exit_code", -1),
                truncated=tc.get("truncated", False),
                result_tokens=tc.get("result_tokens", 0),
                sandbox_id=tc.get("sandbox_id", ""),
            )
            for tc in (tool_calls or [])
        ]

        rollout = GenerationRollout(
            step=step,
            group_id=group_id,
            sample_id=sample_id,
            agent_id=agent_id,
            env=env,
            generations=gen_records,
            env_responses=env_records,
            tool_calls=tc_records,
            reward=reward,
            advantage=advantage,
            sample_metrics=sample_metrics or {},
            golden_answers=golden_answers or {},
            info_turns=info_turns or [],
            sample_tags=sample_tags or {},
            turn_metrics=turn_metrics or [],
            total_tokens=total_tokens,
            raw_string=raw_string,
            compute_reward_time=compute_reward_time,
            stop_reason=stop_reason,
            off_policy_steps=off_policy_steps,
        )

        with self._lock:
            if step not in self._pending_rollouts:
                self._pending_rollouts[step] = []
            self._pending_rollouts[step].append(rollout)
            self._kept_rollouts.append(rollout)

            # Log prompt if this group_id hasn't been logged yet
            if group_id not in self._logged_group_ids:
                self._logged_group_ids.add(group_id)
                prompt_obj = Prompt(
                    step=step,
                    group_id=group_id,
                    env=env,
                    prompt=prompt,
                    tokens_prompt=tokens_prompt,
                    system_prompt=system_prompt,
                    tokens_system_prompt=tokens_system_prompt,
                )
                if step not in self._pending_prompts:
                    self._pending_prompts[step] = []
                self._pending_prompts[step].append(prompt_obj)
                self._kept_prompts.append(prompt_obj)
            self._last_training_step = max(self._last_training_step, step)

    def log_rollout_event(
        self,
        event_type: str,
        phase: str,
        timestamp: float | None = None,
        group_id: int = -1,
        sample_id: int = -1,
        agent_id: int = 0,
        generation_idx: int = -1,
        tool_call_idx: int = -1,
        server_id: int = -1,
        server_lane: int = -1,
        tool_name: str = "",
    ):
        """
        Log a rollout lifecycle event (generation, tool_execution, env_response, reward).

        Args:
            event_type: "generation", "tool_execution", "env_response", "reward"
            phase: "start" or "end"
            timestamp: Event timestamp (defaults to now)
            group_id: Request group ID
            sample_id: Run-wide unique sample ID
            agent_id: Agent ID (0 = main)
            generation_idx: Generation index (-1 if N/A)
            tool_call_idx: Tool call index (-1 if N/A)
            server_id: Inference server index (-1 if N/A)
            server_lane: Per-server lane slot for timeline positioning (-1 if N/A)
            tool_name: Tool name for tool_execution events (inflight display only, not in events table)
        """
        ts = timestamp if timestamp is not None else time.time()
        event = RolloutEvent(
            timestamp=ts,
            event_type=event_type,
            phase=phase,
            group_id=group_id,
            sample_id=sample_id,
            agent_id=agent_id,
            generation_idx=generation_idx,
            tool_call_idx=tool_call_idx,
            server_id=server_id,
            server_lane=server_lane,
        )

        with self._lock:
            self._rollout_events.append(event)

            # Update inflight tracking
            if event_type == "generation":
                if phase == "start" and sample_id >= 0:
                    self._inflight_generations[sample_id] = {
                        "sample_id": sample_id, "generation_idx": generation_idx,
                        "server_id": server_id, "server_lane": server_lane,
                        "group_id": group_id, "agent_id": agent_id,
                        "start_time": ts,
                    }
                elif phase == "end" and sample_id >= 0:
                    self._inflight_generations.pop(sample_id, None)
            elif event_type == "tool_execution":
                if phase == "start" and sample_id >= 0:
                    self._inflight_tool_executions[(sample_id, tool_call_idx)] = {
                        "sample_id": sample_id, "generation_idx": generation_idx,
                        "tool_call_idx": tool_call_idx, "tool_name": tool_name,
                        "agent_id": agent_id,
                    }
                elif phase == "end" and sample_id >= 0:
                    self._inflight_tool_executions.pop((sample_id, tool_call_idx), None)
            elif event_type == "env_response":
                if phase == "start" and sample_id >= 0:
                    self._inflight_env_responses[sample_id] = {
                        "sample_id": sample_id, "generation_idx": generation_idx,
                        "agent_id": agent_id,
                    }
                elif phase == "end" and sample_id >= 0:
                    self._inflight_env_responses.pop(sample_id, None)
            elif event_type == "reward":
                if phase == "start" and sample_id >= 0:
                    self._inflight_rewards[sample_id] = {
                        "sample_id": sample_id, "agent_id": agent_id,
                    }
                elif phase == "end" and sample_id >= 0:
                    self._inflight_rewards.pop(sample_id, None)

    def log_infra_event(
        self,
        event_type: str,
        phase: str,
        timestamp: float | None = None,
        step: int = -1,
        server_id: int = -1,
        sandbox_id: str = "",
    ):
        """
        Log an infrastructure event (weight_sync, sandbox lifecycle).

        Args:
            event_type: "weight_sync" or "sandbox"
            phase: "start"/"end" for weight_sync; "create"/"setup"/"ready"/"execute"/"destroy" for sandbox
            timestamp: Event timestamp (defaults to now)
            step: Training step (-1 if N/A)
            server_id: Server index for weight_sync (-1 if N/A)
            sandbox_id: Sandbox ID for sandbox events
        """
        ts = timestamp if timestamp is not None else time.time()
        event = InfraEvent(
            timestamp=ts,
            event_type=event_type,
            phase=phase,
            step=step,
            server_id=server_id,
            sandbox_id=sandbox_id,
        )

        with self._lock:
            self._infra_events.append(event)

            # Update inflight tracking
            if event_type == "weight_sync":
                if phase == "start" and server_id >= 0:
                    self._inflight_weight_syncs[server_id] = {"server_id": server_id, "step": step}
                elif phase == "end" and server_id >= 0:
                    self._inflight_weight_syncs.pop(server_id, None)
            elif event_type == "sandbox":
                if phase in ("create", "setup"):
                    self._inflight_sandbox_ops[sandbox_id] = {"sandbox_id": sandbox_id, "phase": phase}
                elif phase in ("ready", "destroy"):
                    self._inflight_sandbox_ops.pop(sandbox_id, None)

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

    def log_discarded_generation_rollout(
        self,
        discard_reason: str,
        trainer_step: int,
        inference_step: int,
        group_id: int,
        sample_id: int,
        prompt: str,
        generations: list[dict] | None = None,
        env_responses: list[dict] | None = None,
        tool_calls: list[dict] | None = None,
        reward: float = 0.0,
        advantage: float = 0.0,
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
        stop_reason: str = "",
        off_policy_steps: int = 0,
        agent_id: int = 0,
    ):
        """Log a discarded rollout sample (not sent to trainer)."""
        ts = timestamp if timestamp is not None else time.time()

        gen_records = [
            GenerationRecord(
                generation_idx=g["generation_idx"],
                content=g.get("content", ""),
                tokens=g.get("tokens", 0),
                prompt_tokens=g.get("prompt_tokens", 0),
                tool_call_count=g.get("tool_call_count", 0),
                stop_reason=g.get("stop_reason", ""),
                queue_time=g.get("queue_time", 0.0),
                ttft=g.get("ttft", 0.0),
                prefill_time=g.get("prefill_time", 0.0),
                decode_time=g.get("decode_time", 0.0),
                inference_time=g.get("inference_time", 0.0),
                e2e_latency=g.get("e2e_latency", 0.0),
                server_id=g.get("server_id", -1),
                vllm_request_id=g.get("vllm_request_id", ""),
            )
            for g in (generations or [])
        ]

        env_records = [
            EnvResponseRecord(
                generation_idx=e["generation_idx"],
                content=e.get("content", ""),
                turn_type=e.get("turn_type", "env_response"),
                tokens=e.get("tokens", 0),
                response_time=e.get("response_time", 0.0),
            )
            for e in (env_responses or [])
        ]

        tc_records = [
            ToolCallRecord(
                generation_idx=tc["generation_idx"],
                tool_call_idx=tc.get("tool_call_idx", 0),
                env_response_generation_idx=tc.get("env_response_generation_idx", tc["generation_idx"]),
                tool_name=tc.get("tool_name", ""),
                arguments=tc.get("arguments", ""),
                raw_text=tc.get("raw_text", ""),
                result=tc.get("result", ""),
                success=tc.get("success", True),
                error=tc.get("error", ""),
                exit_code=tc.get("exit_code", -1),
                truncated=tc.get("truncated", False),
                result_tokens=tc.get("result_tokens", 0),
                sandbox_id=tc.get("sandbox_id", ""),
            )
            for tc in (tool_calls or [])
        ]

        rollout = DiscardedGenerationRollout(
            timestamp=ts,
            discard_reason=discard_reason,
            trainer_step=trainer_step,
            inference_step=inference_step,
            group_id=group_id,
            sample_id=sample_id,
            agent_id=agent_id,
            env=env,
            generations=gen_records,
            env_responses=env_records,
            tool_calls=tc_records,
            reward=reward,
            advantage=advantage,
            sample_metrics=sample_metrics or {},
            golden_answers=golden_answers or {},
            info_turns=info_turns or [],
            sample_tags=sample_tags or {},
            total_tokens=total_tokens,
            raw_string=raw_string,
            compute_reward_time=compute_reward_time,
            stop_reason=stop_reason,
            off_policy_steps=off_policy_steps,
        )

        with self._lock:
            self._discarded_rollouts.append(rollout)

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

    def log_cancelled_generation_rollout(
        self,
        cancel_reason: str,
        trainer_step: int,
        inference_step: int,
        group_id: int,
        sample_id: int,
        prompt: str = "",
        env: str = "",
        tokens_prompt: int = 0,
        timestamp: float | None = None,
        off_policy_steps: int = 0,
        agent_id: int = 0,
    ):
        """Log a cancelled rollout sample (inference was interrupted).

        Reuses DiscardedGenerationRollout with empty generation data since
        the inference never completed.
        """
        ts = timestamp if timestamp is not None else time.time()

        rollout = DiscardedGenerationRollout(
            timestamp=ts,
            discard_reason=cancel_reason,
            trainer_step=trainer_step,
            inference_step=inference_step,
            group_id=group_id,
            sample_id=sample_id,
            agent_id=agent_id,
            env=env,
            off_policy_steps=off_policy_steps,
        )

        with self._lock:
            self._cancelled_rollouts.append(rollout)

            if group_id not in self._logged_cancelled_group_ids:
                self._logged_cancelled_group_ids.add(group_id)
                self._cancelled_prompts.append(DiscardedPrompt(
                    timestamp=ts,
                    discard_reason=cancel_reason,
                    trainer_step=trainer_step,
                    inference_step=inference_step,
                    group_id=group_id,
                    env=env,
                    prompt=prompt,
                    tokens_prompt=tokens_prompt,
                ))

    def log_eval_prompt(self, prompt: EvalPrompt):
        """Buffer an eval prompt for upload (deduplicated by step+eval_name+sample_idx)."""
        key = (prompt.step, prompt.eval_name, prompt.sample_idx)
        with self._lock:
            if key not in self._logged_eval_prompt_keys:
                self._logged_eval_prompt_keys.add(key)
                self._eval_prompts.append(prompt)

    def log_eval_rollout(self, rollout: EvalGenerationRollout):
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

    def _rollout_events_to_table(self, events: list[RolloutEvent]) -> pa.Table:
        """Convert list of rollout events to PyArrow table."""
        if not events:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(EVENTS_ROLLOUT_SCHEMA.names, EVENTS_ROLLOUT_SCHEMA)})
        data = {
            "timestamp": [e.timestamp for e in events],
            "tail_idx": [e.tail_idx for e in events],
            "event_type": [e.event_type for e in events],
            "phase": [e.phase for e in events],
            "group_id": [e.group_id for e in events],
            "sample_id": [e.sample_id for e in events],
            "agent_id": [e.agent_id for e in events],
            "generation_idx": [e.generation_idx for e in events],
            "tool_call_idx": [e.tool_call_idx for e in events],
            "server_id": [e.server_id for e in events],
            "server_lane": [e.server_lane for e in events],
        }
        return pa.table(data, schema=EVENTS_ROLLOUT_SCHEMA)

    def _infra_events_to_table(self, events: list[InfraEvent]) -> pa.Table:
        """Convert list of infra events to PyArrow table."""
        if not events:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(EVENTS_INFRA_SCHEMA.names, EVENTS_INFRA_SCHEMA)})
        data = {
            "timestamp": [e.timestamp for e in events],
            "tail_idx": [e.tail_idx for e in events],
            "event_type": [e.event_type for e in events],
            "phase": [e.phase for e in events],
            "step": [e.step for e in events],
            "server_id": [e.server_id for e in events],
            "sandbox_id": [e.sandbox_id for e in events],
        }
        return pa.table(data, schema=EVENTS_INFRA_SCHEMA)

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

    def _thread_pool_metrics_to_table(self, metrics: list[tuple[ThreadPoolMetricSample, int]]) -> pa.Table:
        """Convert list of thread pool metrics (with tail_idx) to PyArrow table."""
        if not metrics:
            return pa.table({
                "timestamp": pa.array([], type=pa.float64()),
                "pool_name": pa.array([], type=pa.string()),
                "metric_name": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "timestamp": [m.timestamp for m, _ in metrics],
            "pool_name": [m.pool_name for m, _ in metrics],
            "metric_name": [m.metric_name for m, _ in metrics],
            "value": [m.value for m, _ in metrics],
            "tail_idx": [tail_idx for _, tail_idx in metrics],
        }
        return pa.table(data, schema=THREAD_POOL_METRICS_SCHEMA)

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

    def _generations_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert generation rollouts to a generations PyArrow table (one row per generation)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(GENERATIONS_SCHEMA.names, GENERATIONS_SCHEMA)})

        steps, group_ids, sample_ids, agent_ids = [], [], [], []
        gen_idxs, contents, tokens_list, prompt_tokens_list = [], [], [], []
        tool_call_counts, stop_reasons = [], []
        queue_times, ttfts, prefill_times, decode_times = [], [], [], []
        inference_times, e2e_latencies, server_ids, vllm_request_ids = [], [], [], []

        for r in rollouts:
            for g in r.generations:
                steps.append(r.step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(g.generation_idx)
                contents.append(_zstd_compress(g.content))
                tokens_list.append(g.tokens)
                prompt_tokens_list.append(g.prompt_tokens)
                tool_call_counts.append(g.tool_call_count)
                stop_reasons.append(g.stop_reason)
                queue_times.append(g.queue_time)
                ttfts.append(g.ttft)
                prefill_times.append(g.prefill_time)
                decode_times.append(g.decode_time)
                inference_times.append(g.inference_time)
                e2e_latencies.append(g.e2e_latency)
                server_ids.append(g.server_id)
                vllm_request_ids.append(g.vllm_request_id)

        return pa.table({
            "step": steps, "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "tokens": tokens_list,
            "prompt_tokens": prompt_tokens_list, "tool_call_count": tool_call_counts,
            "stop_reason": stop_reasons, "queue_time": queue_times, "ttft": ttfts,
            "prefill_time": prefill_times, "decode_time": decode_times,
            "inference_time": inference_times, "e2e_latency": e2e_latencies,
            "server_id": server_ids, "vllm_request_id": vllm_request_ids,
        }, schema=GENERATIONS_SCHEMA)

    def _env_responses_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert generation rollouts to an env_responses PyArrow table (one row per env response)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(ENV_RESPONSES_SCHEMA.names, ENV_RESPONSES_SCHEMA)})

        steps, group_ids, sample_ids, agent_ids = [], [], [], []
        gen_idxs, contents, turn_types, tokens_list, response_times = [], [], [], [], []

        for r in rollouts:
            for e in r.env_responses:
                steps.append(r.step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(e.generation_idx)
                contents.append(_zstd_compress(e.content))
                turn_types.append(e.turn_type)
                tokens_list.append(e.tokens)
                response_times.append(e.response_time)

        return pa.table({
            "step": steps, "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "turn_type": turn_types,
            "tokens": tokens_list, "response_time": response_times,
        }, schema=ENV_RESPONSES_SCHEMA)

    def _tool_calls_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert generation rollouts to a tool_calls PyArrow table (one row per tool call)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(TOOL_CALLS_SCHEMA.names, TOOL_CALLS_SCHEMA)})

        steps, group_ids, sample_ids, agent_ids = [], [], [], []
        gen_idxs, tc_idxs, env_resp_gen_idxs = [], [], []
        tool_names, arguments_list, raw_texts, results = [], [], [], []
        successes, errors, exit_codes, truncateds = [], [], [], []
        result_tokens_list, sandbox_ids = [], []

        for r in rollouts:
            for tc in r.tool_calls:
                steps.append(r.step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(tc.generation_idx)
                tc_idxs.append(tc.tool_call_idx)
                env_resp_gen_idxs.append(tc.env_response_generation_idx)
                tool_names.append(tc.tool_name)
                arguments_list.append(tc.arguments)
                raw_texts.append(tc.raw_text)
                results.append(_zstd_compress(tc.result))
                successes.append(tc.success)
                errors.append(tc.error)
                exit_codes.append(tc.exit_code)
                truncateds.append(tc.truncated)
                result_tokens_list.append(tc.result_tokens)
                sandbox_ids.append(tc.sandbox_id)

        return pa.table({
            "step": steps, "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env_response_generation_idx": env_resp_gen_idxs,
            "tool_name": tool_names, "arguments": arguments_list, "raw_text": raw_texts,
            "result": results, "success": successes, "error": errors, "exit_code": exit_codes,
            "truncated": truncateds, "result_tokens": result_tokens_list, "sandbox_id": sandbox_ids,
        }, schema=TOOL_CALLS_SCHEMA)

    def _turn_metrics_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert generation rollouts to a turn_metrics PyArrow table."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(TURN_METRICS_SCHEMA.names, TURN_METRICS_SCHEMA)})

        steps, sample_ids, agent_ids = [], [], []
        gen_idxs, turn_types, envs, metric_names, values = [], [], [], [], []

        for r in rollouts:
            for tm in r.turn_metrics:
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(tm.get("generation_idx", 0))
                turn_types.append(tm.get("turn_type", "generation"))
                envs.append(r.env)
                metric_names.append(tm.get("metric_name", ""))
                values.append(float(tm.get("value", 0.0)))

        return pa.table({
            "step": steps, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "turn_type": turn_types, "env": envs,
            "metric_name": metric_names, "value": values,
        }, schema=TURN_METRICS_SCHEMA)

    def _samples_data_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert generation rollouts to a samples_data table (one row per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(SAMPLES_DATA_SCHEMA.names, SAMPLES_DATA_SCHEMA)})
        return pa.table({
            "step": [r.step for r in rollouts],
            "group_id": [r.group_id for r in rollouts],
            "sample_id": [r.sample_id for r in rollouts],
            "reward": [r.reward for r in rollouts],
            "advantage": [r.advantage for r in rollouts],
            "num_generations": [len(r.generations) for r in rollouts],
            "total_tokens": [r.total_tokens for r in rollouts],
            "raw_string": [_zstd_compress(r.raw_string) for r in rollouts],
            "compute_reward_time": [r.compute_reward_time for r in rollouts],
            "stop_reason": [r.stop_reason for r in rollouts],
            "off_policy_steps": [r.off_policy_steps for r in rollouts],
        }, schema=SAMPLES_DATA_SCHEMA)

    def _rollouts_metrics_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert rollouts' sample_metrics to normalized table (one row per metric per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(ROLLOUTS_METRICS_SCHEMA.names, ROLLOUTS_METRICS_SCHEMA)})
        steps, sample_ids, envs, metric_names, values = [], [], [], [], []
        for r in rollouts:
            for metric_name, value in r.sample_metrics.items():
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                metric_names.append(metric_name)
                values.append(float(value))
        return pa.table({"step": steps, "sample_id": sample_ids, "env": envs, "metric_name": metric_names, "value": values}, schema=ROLLOUTS_METRICS_SCHEMA)

    def _golden_answers_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert rollouts' golden_answers to normalized table (one row per answer per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(GOLDEN_ANSWERS_SCHEMA.names, GOLDEN_ANSWERS_SCHEMA)})
        steps, sample_ids, envs, keys, values = [], [], [], [], []
        for r in rollouts:
            for key, value in r.golden_answers.items():
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                keys.append(key)
                values.append(value)
        return pa.table({"step": steps, "sample_id": sample_ids, "env": envs, "key": keys, "value": values}, schema=GOLDEN_ANSWERS_SCHEMA)

    def _info_turns_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert rollouts' info_turns to normalized table (one row per info item)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(INFO_TURNS_SCHEMA.names, INFO_TURNS_SCHEMA)})
        steps, sample_ids, agent_ids, gen_idxs, tc_idxs = [], [], [], [], []
        envs, info_keys, info_values, info_types = [], [], [], []
        for r in rollouts:
            for info in r.info_turns:
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(info.get("generation_idx", info.get("turn_order", 0)))
                tc_idxs.append(info.get("tool_call_idx", -1))
                envs.append(r.env)
                info_keys.append(info.get("info_key", ""))
                info_values.append(info.get("info_value", ""))
                info_types.append(info.get("info_type", "text"))
        return pa.table({
            "step": steps, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env": envs, "info_key": info_keys, "info_value": info_values, "info_type": info_types,
        }, schema=INFO_TURNS_SCHEMA)

    def _sample_tags_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert rollouts' sample_tags to normalized table (one row per tag per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(SAMPLE_TAGS_SCHEMA.names, SAMPLE_TAGS_SCHEMA)})
        steps, sample_ids, envs, tag_names, tag_values = [], [], [], [], []
        for r in rollouts:
            for tag_name, tag_value in r.sample_tags.items():
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                tag_names.append(tag_name)
                tag_values.append(str(tag_value))
        return pa.table({"step": steps, "sample_id": sample_ids, "env": envs, "tag_name": tag_names, "tag_value": tag_values}, schema=SAMPLE_TAGS_SCHEMA)

    def _kept_prompts_to_table(self, prompts: list[Prompt]) -> pa.Table:
        """Convert list of kept prompts to PyArrow table for events system."""
        if not prompts:
            return pa.table({
                "step": pa.array([], type=pa.int32()),
                "group_id": pa.array([], type=pa.int32()),
                "env": pa.array([], type=pa.string()),
                "prompt": pa.array([], type=pa.string()),
                "tokens_prompt": pa.array([], type=pa.int32()),
                "system_prompt": pa.array([], type=pa.string()),
                "tokens_system_prompt": pa.array([], type=pa.int32()),
                "tail_idx": pa.array([], type=pa.int32()),
            })

        data = {
            "step": [p.step for p in prompts],
            "group_id": [p.group_id for p in prompts],
            "env": [p.env for p in prompts],
            "prompt": [p.prompt for p in prompts],
            "tokens_prompt": [p.tokens_prompt for p in prompts],
            "system_prompt": [p.system_prompt for p in prompts],
            "tokens_system_prompt": [p.tokens_system_prompt for p in prompts],
            "tail_idx": [p.tail_idx for p in prompts],
        }
        return pa.table(data)

    def _kept_generations_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts to a generations PyArrow table (one row per generation, with tail_idx)."""
        schema = pa.schema([*GENERATIONS_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})

        steps, group_ids, sample_ids, agent_ids = [], [], [], []
        gen_idxs, contents, tokens_list, prompt_tokens_list = [], [], [], []
        tool_call_counts, stop_reasons = [], []
        queue_times, ttfts, prefill_times, decode_times = [], [], [], []
        inference_times, e2e_latencies, server_ids, vllm_request_ids = [], [], [], []
        tail_idxs = []

        for r in rollouts:
            for g in r.generations:
                steps.append(r.step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(g.generation_idx)
                contents.append(_zstd_compress(g.content))
                tokens_list.append(g.tokens)
                prompt_tokens_list.append(g.prompt_tokens)
                tool_call_counts.append(g.tool_call_count)
                stop_reasons.append(g.stop_reason)
                queue_times.append(g.queue_time)
                ttfts.append(g.ttft)
                prefill_times.append(g.prefill_time)
                decode_times.append(g.decode_time)
                inference_times.append(g.inference_time)
                e2e_latencies.append(g.e2e_latency)
                server_ids.append(g.server_id)
                vllm_request_ids.append(g.vllm_request_id)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "tokens": tokens_list,
            "prompt_tokens": prompt_tokens_list, "tool_call_count": tool_call_counts,
            "stop_reason": stop_reasons, "queue_time": queue_times, "ttft": ttfts,
            "prefill_time": prefill_times, "decode_time": decode_times,
            "inference_time": inference_times, "e2e_latency": e2e_latencies,
            "server_id": server_ids, "vllm_request_id": vllm_request_ids,
            "tail_idx": tail_idxs,
        }, schema=schema)

    def _kept_env_responses_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts to an env_responses PyArrow table (one row per env response, with tail_idx)."""
        schema = pa.schema([*ENV_RESPONSES_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})

        steps, group_ids, sample_ids, agent_ids = [], [], [], []
        gen_idxs, contents, turn_types, tokens_list, response_times = [], [], [], [], []
        tail_idxs = []

        for r in rollouts:
            for e in r.env_responses:
                steps.append(r.step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(e.generation_idx)
                contents.append(_zstd_compress(e.content))
                turn_types.append(e.turn_type)
                tokens_list.append(e.tokens)
                response_times.append(e.response_time)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "turn_type": turn_types,
            "tokens": tokens_list, "response_time": response_times,
            "tail_idx": tail_idxs,
        }, schema=schema)

    def _kept_tool_calls_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts to a tool_calls PyArrow table (one row per tool call, with tail_idx)."""
        schema = pa.schema([*TOOL_CALLS_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})

        steps, group_ids, sample_ids, agent_ids = [], [], [], []
        gen_idxs, tc_idxs, env_resp_gen_idxs = [], [], []
        tool_names, arguments_list, raw_texts, results = [], [], [], []
        successes, errors, exit_codes, truncateds = [], [], [], []
        result_tokens_list, sandbox_ids = [], []
        tail_idxs = []

        for r in rollouts:
            for tc in r.tool_calls:
                steps.append(r.step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(tc.generation_idx)
                tc_idxs.append(tc.tool_call_idx)
                env_resp_gen_idxs.append(tc.env_response_generation_idx)
                tool_names.append(tc.tool_name)
                arguments_list.append(tc.arguments)
                raw_texts.append(tc.raw_text)
                results.append(_zstd_compress(tc.result))
                successes.append(tc.success)
                errors.append(tc.error)
                exit_codes.append(tc.exit_code)
                truncateds.append(tc.truncated)
                result_tokens_list.append(tc.result_tokens)
                sandbox_ids.append(tc.sandbox_id)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env_response_generation_idx": env_resp_gen_idxs,
            "tool_name": tool_names, "arguments": arguments_list, "raw_text": raw_texts,
            "result": results, "success": successes, "error": errors, "exit_code": exit_codes,
            "truncated": truncateds, "result_tokens": result_tokens_list, "sandbox_id": sandbox_ids,
            "tail_idx": tail_idxs,
        }, schema=schema)

    def _kept_samples_data_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts to a samples_data table for events system."""
        schema = pa.schema([*SAMPLES_DATA_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})
        return pa.table({
            "step": [r.step for r in rollouts],
            "group_id": [r.group_id for r in rollouts],
            "sample_id": [r.sample_id for r in rollouts],
            "reward": [r.reward for r in rollouts],
            "advantage": [r.advantage for r in rollouts],
            "num_generations": [len(r.generations) for r in rollouts],
            "total_tokens": [r.total_tokens for r in rollouts],
            "raw_string": [_zstd_compress(r.raw_string) for r in rollouts],
            "compute_reward_time": [r.compute_reward_time for r in rollouts],
            "stop_reason": [r.stop_reason for r in rollouts],
            "off_policy_steps": [r.off_policy_steps for r in rollouts],
            "tail_idx": [r.tail_idx for r in rollouts],
        }, schema=schema)

    def _kept_rollouts_metrics_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts' metrics to a normalized table for events system."""
        schema = pa.schema([*ROLLOUTS_METRICS_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})

        steps, sample_ids, envs, metric_names, values, tail_idxs = [], [], [], [], [], []
        for r in rollouts:
            for metric_name, value in r.sample_metrics.items():
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                metric_names.append(metric_name)
                values.append(float(value))
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "sample_id": sample_ids, "env": envs,
            "metric_name": metric_names, "value": values, "tail_idx": tail_idxs,
        }, schema=schema)

    def _kept_golden_answers_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts' golden answers to a normalized table for events system."""
        schema = pa.schema([*GOLDEN_ANSWERS_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})

        steps, sample_ids, envs, keys, values, tail_idxs = [], [], [], [], [], []
        for r in rollouts:
            for key, value in r.golden_answers.items():
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                keys.append(key)
                values.append(value)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "sample_id": sample_ids, "env": envs,
            "key": keys, "value": values, "tail_idx": tail_idxs,
        }, schema=schema)

    def _kept_info_turns_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts' info_turns to a normalized table for events system."""
        schema = pa.schema([*INFO_TURNS_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})

        steps, sample_ids, agent_ids, gen_idxs, tc_idxs = [], [], [], [], []
        envs, info_keys, info_values, info_types, tail_idxs = [], [], [], [], []
        for r in rollouts:
            for info in r.info_turns:
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(info.get("generation_idx", info.get("turn_order", 0)))
                tc_idxs.append(info.get("tool_call_idx", -1))
                envs.append(r.env)
                info_keys.append(info.get("info_key", ""))
                info_values.append(info.get("info_value", ""))
                info_types.append(info.get("info_type", "text"))
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env": envs, "info_key": info_keys, "info_value": info_values, "info_type": info_types,
            "tail_idx": tail_idxs,
        }, schema=schema)

    def _kept_sample_tags_to_table(self, rollouts: list[GenerationRollout]) -> pa.Table:
        """Convert kept rollouts' sample_tags to a normalized table for events system."""
        schema = pa.schema([*SAMPLE_TAGS_SCHEMA, pa.field("tail_idx", pa.int32())])
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(schema.names, schema)})

        steps, sample_ids, envs, tag_names, tag_values, tail_idxs = [], [], [], [], [], []
        for r in rollouts:
            for tag_name, tag_value in r.sample_tags.items():
                steps.append(r.step)
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                tag_names.append(tag_name)
                tag_values.append(str(tag_value))
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "sample_id": sample_ids, "env": envs,
            "tag_name": tag_names, "tag_value": tag_values, "tail_idx": tail_idxs,
        }, schema=schema)

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

    def _discarded_generations_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts to a generations table (one row per generation)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(GENERATIONS_DISCARDED_SCHEMA.names, GENERATIONS_DISCARDED_SCHEMA)})

        trainer_steps, inference_steps, group_ids, sample_ids, agent_ids = [], [], [], [], []
        gen_idxs, contents, tokens_list, prompt_tokens_list = [], [], [], []
        tool_call_counts, stop_reasons = [], []
        queue_times, ttfts, prefill_times, decode_times = [], [], [], []
        inference_times, e2e_latencies, server_ids, vllm_request_ids = [], [], [], []
        tail_idxs = []

        for r in rollouts:
            for g in r.generations:
                trainer_steps.append(r.trainer_step)
                inference_steps.append(r.inference_step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(g.generation_idx)
                contents.append(_zstd_compress(g.content))
                tokens_list.append(g.tokens)
                prompt_tokens_list.append(g.prompt_tokens)
                tool_call_counts.append(g.tool_call_count)
                stop_reasons.append(g.stop_reason)
                queue_times.append(g.queue_time)
                ttfts.append(g.ttft)
                prefill_times.append(g.prefill_time)
                decode_times.append(g.decode_time)
                inference_times.append(g.inference_time)
                e2e_latencies.append(g.e2e_latency)
                server_ids.append(g.server_id)
                vllm_request_ids.append(g.vllm_request_id)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "trainer_step": trainer_steps, "inference_step": inference_steps,
            "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "tokens": tokens_list,
            "prompt_tokens": prompt_tokens_list, "tool_call_count": tool_call_counts,
            "stop_reason": stop_reasons, "queue_time": queue_times, "ttft": ttfts,
            "prefill_time": prefill_times, "decode_time": decode_times,
            "inference_time": inference_times, "e2e_latency": e2e_latencies,
            "server_id": server_ids, "vllm_request_id": vllm_request_ids,
            "tail_idx": tail_idxs,
        }, schema=GENERATIONS_DISCARDED_SCHEMA)

    def _discarded_env_responses_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts to an env_responses table (one row per env response)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(ENV_RESPONSES_DISCARDED_SCHEMA.names, ENV_RESPONSES_DISCARDED_SCHEMA)})

        trainer_steps, inference_steps, group_ids, sample_ids, agent_ids = [], [], [], [], []
        gen_idxs, contents, turn_types, tokens_list, response_times = [], [], [], [], []
        tail_idxs = []

        for r in rollouts:
            for e in r.env_responses:
                trainer_steps.append(r.trainer_step)
                inference_steps.append(r.inference_step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(e.generation_idx)
                contents.append(_zstd_compress(e.content))
                turn_types.append(e.turn_type)
                tokens_list.append(e.tokens)
                response_times.append(e.response_time)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "trainer_step": trainer_steps, "inference_step": inference_steps,
            "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "turn_type": turn_types,
            "tokens": tokens_list, "response_time": response_times,
            "tail_idx": tail_idxs,
        }, schema=ENV_RESPONSES_DISCARDED_SCHEMA)

    def _discarded_tool_calls_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts to a tool_calls table (one row per tool call)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(TOOL_CALLS_DISCARDED_SCHEMA.names, TOOL_CALLS_DISCARDED_SCHEMA)})

        trainer_steps, inference_steps, group_ids, sample_ids, agent_ids = [], [], [], [], []
        gen_idxs, tc_idxs, env_resp_gen_idxs = [], [], []
        tool_names, arguments_list, raw_texts, results = [], [], [], []
        successes, errors, exit_codes, truncateds = [], [], [], []
        result_tokens_list, sandbox_ids = [], []
        tail_idxs = []

        for r in rollouts:
            for tc in r.tool_calls:
                trainer_steps.append(r.trainer_step)
                inference_steps.append(r.inference_step)
                group_ids.append(r.group_id)
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(tc.generation_idx)
                tc_idxs.append(tc.tool_call_idx)
                env_resp_gen_idxs.append(tc.env_response_generation_idx)
                tool_names.append(tc.tool_name)
                arguments_list.append(tc.arguments)
                raw_texts.append(tc.raw_text)
                results.append(_zstd_compress(tc.result))
                successes.append(tc.success)
                errors.append(tc.error)
                exit_codes.append(tc.exit_code)
                truncateds.append(tc.truncated)
                result_tokens_list.append(tc.result_tokens)
                sandbox_ids.append(tc.sandbox_id)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "trainer_step": trainer_steps, "inference_step": inference_steps,
            "group_id": group_ids, "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env_response_generation_idx": env_resp_gen_idxs,
            "tool_name": tool_names, "arguments": arguments_list, "raw_text": raw_texts,
            "result": results, "success": successes, "error": errors, "exit_code": exit_codes,
            "truncated": truncateds, "result_tokens": result_tokens_list, "sandbox_id": sandbox_ids,
            "tail_idx": tail_idxs,
        }, schema=TOOL_CALLS_DISCARDED_SCHEMA)

    def _samples_data_discarded_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts to a samples_data table (one row per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(SAMPLES_DATA_DISCARDED_SCHEMA.names, SAMPLES_DATA_DISCARDED_SCHEMA)})
        return pa.table({
            "timestamp": [r.timestamp for r in rollouts],
            "discard_reason": [r.discard_reason for r in rollouts],
            "trainer_step": [r.trainer_step for r in rollouts],
            "inference_step": [r.inference_step for r in rollouts],
            "group_id": [r.group_id for r in rollouts],
            "sample_id": [r.sample_id for r in rollouts],
            "reward": [r.reward for r in rollouts],
            "advantage": [r.advantage for r in rollouts],
            "num_generations": [len(r.generations) for r in rollouts],
            "total_tokens": [r.total_tokens for r in rollouts],
            "raw_string": [_zstd_compress(r.raw_string) for r in rollouts],
            "compute_reward_time": [r.compute_reward_time for r in rollouts],
            "stop_reason": [r.stop_reason for r in rollouts],
            "off_policy_steps": [r.off_policy_steps for r in rollouts],
            "tail_idx": [r.tail_idx for r in rollouts],
        }, schema=SAMPLES_DATA_DISCARDED_SCHEMA)

    def _rollouts_metrics_discarded_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts' metrics to a normalized table (one row per metric per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(ROLLOUTS_METRICS_DISCARDED_SCHEMA.names, ROLLOUTS_METRICS_DISCARDED_SCHEMA)})

        sample_ids, envs, metric_names, values, tail_idxs = [], [], [], [], []
        for r in rollouts:
            for metric_name, value in r.sample_metrics.items():
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                metric_names.append(metric_name)
                values.append(float(value))
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "sample_id": sample_ids, "env": envs, "metric_name": metric_names,
            "value": values, "tail_idx": tail_idxs,
        }, schema=ROLLOUTS_METRICS_DISCARDED_SCHEMA)

    def _golden_answers_discarded_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts' golden answers to a normalized table (one row per answer per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(GOLDEN_ANSWERS_DISCARDED_SCHEMA.names, GOLDEN_ANSWERS_DISCARDED_SCHEMA)})

        sample_ids, envs, keys, values, tail_idxs = [], [], [], [], []
        for r in rollouts:
            for key, value in r.golden_answers.items():
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                keys.append(key)
                values.append(value)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "sample_id": sample_ids, "env": envs, "key": keys,
            "value": values, "tail_idx": tail_idxs,
        }, schema=GOLDEN_ANSWERS_DISCARDED_SCHEMA)

    def _info_turns_discarded_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts' info_turns to a normalized table (one row per info item)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(INFO_TURNS_DISCARDED_SCHEMA.names, INFO_TURNS_DISCARDED_SCHEMA)})

        sample_ids, agent_ids, gen_idxs, tc_idxs = [], [], [], []
        envs, info_keys, info_values, info_types, tail_idxs = [], [], [], [], []
        for r in rollouts:
            for info in r.info_turns:
                sample_ids.append(r.sample_id)
                agent_ids.append(r.agent_id)
                gen_idxs.append(info.get("generation_idx", info.get("turn_order", 0)))
                tc_idxs.append(info.get("tool_call_idx", -1))
                envs.append(r.env)
                info_keys.append(info.get("info_key", ""))
                info_values.append(info.get("info_value", ""))
                info_types.append(info.get("info_type", "text"))
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "sample_id": sample_ids, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env": envs, "info_key": info_keys, "info_value": info_values, "info_type": info_types,
            "tail_idx": tail_idxs,
        }, schema=INFO_TURNS_DISCARDED_SCHEMA)

    def _sample_tags_discarded_to_table(self, rollouts: list[DiscardedGenerationRollout]) -> pa.Table:
        """Convert discarded rollouts' sample_tags to a normalized table (one row per tag per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(SAMPLE_TAGS_DISCARDED_SCHEMA.names, SAMPLE_TAGS_DISCARDED_SCHEMA)})

        sample_ids, envs, tag_names, tag_values, tail_idxs = [], [], [], [], []
        for r in rollouts:
            for tag_name, tag_value in r.sample_tags.items():
                sample_ids.append(r.sample_id)
                envs.append(r.env)
                tag_names.append(tag_name)
                tag_values.append(str(tag_value))
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "sample_id": sample_ids, "env": envs, "tag_name": tag_names,
            "tag_value": tag_values, "tail_idx": tail_idxs,
        }, schema=SAMPLE_TAGS_DISCARDED_SCHEMA)

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
            "sample_id": [p.sample_id for p in prompts],
            "env": [p.env for p in prompts],
            "prompt": [p.prompt for p in prompts],
            "tokens_prompt": [p.tokens_prompt for p in prompts],
            "system_prompt": [p.system_prompt for p in prompts],
            "tokens_system_prompt": [p.tokens_system_prompt for p in prompts],
            "tail_idx": [p.tail_idx for p in prompts],
        }
        return pa.table(data, schema=PROMPTS_EVAL_SCHEMA)

    def _eval_generations_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts to a generations table (one row per generation)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(GENERATIONS_EVAL_SCHEMA.names, GENERATIONS_EVAL_SCHEMA)})

        steps, eval_names, model_steps, sample_idxs, sample_ids, comp_idxs, agent_ids = [], [], [], [], [], [], []
        gen_idxs, contents, tokens_list, prompt_tokens_list = [], [], [], []
        tool_call_counts, stop_reasons, tail_idxs = [], [], []

        for r in rollouts:
            for g in r.generations:
                steps.append(r.step)
                eval_names.append(r.eval_name)
                model_steps.append(r.model_step)
                sample_idxs.append(r.sample_idx)
                sample_ids.append(r.sample_id)
                comp_idxs.append(r.completion_idx)
                agent_ids.append(r.agent_id)
                gen_idxs.append(g.generation_idx)
                contents.append(_zstd_compress(g.content))
                tokens_list.append(g.tokens)
                prompt_tokens_list.append(g.prompt_tokens)
                tool_call_counts.append(g.tool_call_count)
                stop_reasons.append(g.stop_reason)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "eval_name": eval_names, "model_step": model_steps,
            "sample_idx": sample_idxs, "sample_id": sample_ids,
            "completion_idx": comp_idxs, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "tokens": tokens_list,
            "prompt_tokens": prompt_tokens_list, "tool_call_count": tool_call_counts,
            "stop_reason": stop_reasons, "tail_idx": tail_idxs,
        }, schema=GENERATIONS_EVAL_SCHEMA)

    def _eval_env_responses_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts to an env_responses table (one row per env response)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(ENV_RESPONSES_EVAL_SCHEMA.names, ENV_RESPONSES_EVAL_SCHEMA)})

        steps, eval_names, model_steps, sample_idxs, sample_ids, comp_idxs, agent_ids = [], [], [], [], [], [], []
        gen_idxs, contents, turn_types, tokens_list, response_times, tail_idxs = [], [], [], [], [], []

        for r in rollouts:
            for e in r.env_responses:
                steps.append(r.step)
                eval_names.append(r.eval_name)
                model_steps.append(r.model_step)
                sample_idxs.append(r.sample_idx)
                sample_ids.append(r.sample_id)
                comp_idxs.append(r.completion_idx)
                agent_ids.append(r.agent_id)
                gen_idxs.append(e.generation_idx)
                contents.append(_zstd_compress(e.content))
                turn_types.append(e.turn_type)
                tokens_list.append(e.tokens)
                response_times.append(e.response_time)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "eval_name": eval_names, "model_step": model_steps,
            "sample_idx": sample_idxs, "sample_id": sample_ids,
            "completion_idx": comp_idxs, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "content": contents, "turn_type": turn_types,
            "tokens": tokens_list, "response_time": response_times, "tail_idx": tail_idxs,
        }, schema=ENV_RESPONSES_EVAL_SCHEMA)

    def _eval_tool_calls_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts to a tool_calls table (one row per tool call)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(TOOL_CALLS_EVAL_SCHEMA.names, TOOL_CALLS_EVAL_SCHEMA)})

        steps, eval_names, model_steps, sample_idxs, sample_ids, comp_idxs, agent_ids = [], [], [], [], [], [], []
        gen_idxs, tc_idxs, env_resp_gen_idxs = [], [], []
        tool_names, arguments_list, raw_texts, results = [], [], [], []
        successes, errors, exit_codes, truncateds = [], [], [], []
        result_tokens_list, sandbox_ids, tail_idxs = [], [], []

        for r in rollouts:
            for tc in r.tool_calls:
                steps.append(r.step)
                eval_names.append(r.eval_name)
                model_steps.append(r.model_step)
                sample_idxs.append(r.sample_idx)
                sample_ids.append(r.sample_id)
                comp_idxs.append(r.completion_idx)
                agent_ids.append(r.agent_id)
                gen_idxs.append(tc.generation_idx)
                tc_idxs.append(tc.tool_call_idx)
                env_resp_gen_idxs.append(tc.env_response_generation_idx)
                tool_names.append(tc.tool_name)
                arguments_list.append(tc.arguments)
                raw_texts.append(tc.raw_text)
                results.append(_zstd_compress(tc.result))
                successes.append(tc.success)
                errors.append(tc.error)
                exit_codes.append(tc.exit_code)
                truncateds.append(tc.truncated)
                result_tokens_list.append(tc.result_tokens)
                sandbox_ids.append(tc.sandbox_id)
                tail_idxs.append(r.tail_idx)

        return pa.table({
            "step": steps, "eval_name": eval_names, "model_step": model_steps,
            "sample_idx": sample_idxs, "sample_id": sample_ids,
            "completion_idx": comp_idxs, "agent_id": agent_ids,
            "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env_response_generation_idx": env_resp_gen_idxs,
            "tool_name": tool_names, "arguments": arguments_list, "raw_text": raw_texts,
            "result": results, "success": successes, "error": errors, "exit_code": exit_codes,
            "truncated": truncateds, "result_tokens": result_tokens_list, "sandbox_id": sandbox_ids,
            "tail_idx": tail_idxs,
        }, schema=TOOL_CALLS_EVAL_SCHEMA)

    def _samples_data_eval_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts to a samples_data table (one row per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(SAMPLES_DATA_EVAL_SCHEMA.names, SAMPLES_DATA_EVAL_SCHEMA)})
        return pa.table({
            "step": [r.step for r in rollouts],
            "eval_name": [r.eval_name for r in rollouts],
            "model_step": [r.model_step for r in rollouts],
            "sample_idx": [r.sample_idx for r in rollouts],
            "sample_id": [r.sample_id for r in rollouts],
            "completion_idx": [r.completion_idx for r in rollouts],
            "env": [r.env for r in rollouts],
            "num_generations": [len(r.generations) for r in rollouts],
            "compute_eval_metrics_time": [r.compute_eval_metrics_time for r in rollouts],
            "stop_reason": [r.stop_reason for r in rollouts],
            "tail_idx": [r.tail_idx for r in rollouts],
        }, schema=SAMPLES_DATA_EVAL_SCHEMA)

    def _rollouts_metrics_eval_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts' metrics to a normalized table (one row per metric per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(ROLLOUTS_METRICS_EVAL_SCHEMA.names, ROLLOUTS_METRICS_EVAL_SCHEMA)})
        steps, eval_names, sample_idxs, sample_ids, comp_idxs, envs, metric_names, values, tail_idxs = [], [], [], [], [], [], [], [], []
        for r in rollouts:
            for metric_name, value in r.sample_metrics.items():
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                sample_ids.append(r.sample_id)
                comp_idxs.append(r.completion_idx)
                envs.append(r.env)
                metric_names.append(metric_name)
                values.append(float(value))
                tail_idxs.append(r.tail_idx)
        return pa.table({
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "sample_id": sample_ids,
            "completion_idx": comp_idxs,
            "env": envs, "metric_name": metric_names,
            "value": values, "tail_idx": tail_idxs,
        }, schema=ROLLOUTS_METRICS_EVAL_SCHEMA)

    def _golden_answers_eval_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts' golden answers to a normalized table (one row per answer per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(GOLDEN_ANSWERS_EVAL_SCHEMA.names, GOLDEN_ANSWERS_EVAL_SCHEMA)})
        steps, eval_names, sample_idxs, sample_ids, comp_idxs, envs, keys, values, tail_idxs = [], [], [], [], [], [], [], [], []
        for r in rollouts:
            for key, value in r.golden_answers.items():
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                sample_ids.append(r.sample_id)
                comp_idxs.append(r.completion_idx)
                envs.append(r.env)
                keys.append(key)
                values.append(value)
                tail_idxs.append(r.tail_idx)
        return pa.table({
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "sample_id": sample_ids,
            "completion_idx": comp_idxs,
            "env": envs, "key": keys, "value": values, "tail_idx": tail_idxs,
        }, schema=GOLDEN_ANSWERS_EVAL_SCHEMA)

    def _info_turns_eval_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts' info_turns to a normalized table (one row per info item)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(INFO_TURNS_EVAL_SCHEMA.names, INFO_TURNS_EVAL_SCHEMA)})
        steps, eval_names, sample_idxs, sample_ids, comp_idxs, agent_ids = [], [], [], [], [], []
        gen_idxs, tc_idxs, envs = [], [], []
        info_keys, info_values, info_types, tail_idxs = [], [], [], []
        for r in rollouts:
            for info in r.info_turns:
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                sample_ids.append(r.sample_id)
                comp_idxs.append(r.completion_idx)
                agent_ids.append(r.agent_id)
                gen_idxs.append(info.get("generation_idx", info.get("turn_order", 0)))
                tc_idxs.append(info.get("tool_call_idx", -1))
                envs.append(r.env)
                info_keys.append(info.get("info_key", ""))
                info_values.append(info.get("info_value", ""))
                info_types.append(info.get("info_type", "text"))
                tail_idxs.append(r.tail_idx)
        return pa.table({
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "sample_id": sample_ids,
            "completion_idx": comp_idxs,
            "agent_id": agent_ids, "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
            "env": envs, "info_key": info_keys, "info_value": info_values,
            "info_type": info_types, "tail_idx": tail_idxs,
        }, schema=INFO_TURNS_EVAL_SCHEMA)

    def _sample_tags_eval_to_table(self, rollouts: list[EvalGenerationRollout]) -> pa.Table:
        """Convert eval rollouts' sample_tags to a normalized table (one row per tag per sample)."""
        if not rollouts:
            return pa.table({col: pa.array([], type=field.type) for col, field in zip(SAMPLE_TAGS_EVAL_SCHEMA.names, SAMPLE_TAGS_EVAL_SCHEMA)})
        steps, eval_names, sample_idxs, sample_ids, comp_idxs, envs, tag_names, tag_values, tail_idxs = [], [], [], [], [], [], [], [], []
        for r in rollouts:
            for tag_name, tag_value in r.sample_tags.items():
                steps.append(r.step)
                eval_names.append(r.eval_name)
                sample_idxs.append(r.sample_idx)
                sample_ids.append(r.sample_id)
                comp_idxs.append(r.completion_idx)
                envs.append(r.env)
                tag_names.append(tag_name)
                tag_values.append(str(tag_value))
                tail_idxs.append(r.tail_idx)
        return pa.table({
            "step": steps, "eval_name": eval_names,
            "sample_idx": sample_idxs, "sample_id": sample_ids,
            "completion_idx": comp_idxs,
            "env": envs, "tag_name": tag_names,
            "tag_value": tag_values, "tail_idx": tail_idxs,
        }, schema=SAMPLE_TAGS_EVAL_SCHEMA)

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
        rollout_events: list[RolloutEvent],
        infra_events: list[InfraEvent],
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
            [e.tail_idx for e in rollout_events] +
            [e.tail_idx for e in infra_events] +
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
        rollout_events: list[RolloutEvent],
        infra_events: list[InfraEvent],
        gpu_metrics: list,
        cpu_metrics: list,
        vllm_metrics: list,
        path: str,
        metadata: dict | None = None,
        discarded_rollouts: list[DiscardedGenerationRollout] | None = None,
        discarded_prompts: list[DiscardedPrompt] | None = None,
        kept_rollouts: list[GenerationRollout] | None = None,
        kept_prompts: list[Prompt] | None = None,
        eval_rollouts: list[EvalGenerationRollout] | None = None,
        eval_prompts: list[EvalPrompt] | None = None,
        cancelled_rollouts: list[DiscardedGenerationRollout] | None = None,
        cancelled_prompts: list[DiscardedPrompt] | None = None,
        logs: list[tuple[LogRecord, int]] | None = None,
        inflight_snapshot: dict | None = None,
        thread_pool_metrics: list | None = None,
    ):
        """Write all event data as separate parquet files in a single zip archive."""
        if self.run is None:
            return

        dest_path = Path(self.run.dir) / path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert all data to tables
        orchestrator_table = self._orchestrator_events_to_table(orchestrator_events)
        trainer_table = self._trainer_events_to_table(trainer_events)
        rollout_events_table = self._rollout_events_to_table(rollout_events)
        infra_events_table = self._infra_events_to_table(infra_events)
        gpu_table = self._gpu_metrics_to_table(gpu_metrics)
        cpu_table = self._cpu_metrics_to_table(cpu_metrics)
        vllm_table = self._vllm_metrics_to_table(vllm_metrics)
        thread_pool_table = self._thread_pool_metrics_to_table(thread_pool_metrics or [])

        # Convert discarded prompts and rollouts (empty lists if None)
        discarded_proms = discarded_prompts or []
        discarded_gens = discarded_rollouts or []
        discarded_prompts_table = self._discarded_prompts_to_table(discarded_proms)
        discarded_gen_table = self._discarded_generations_to_table(discarded_gens)
        discarded_env_resp_table = self._discarded_env_responses_to_table(discarded_gens)
        discarded_tool_calls_table = self._discarded_tool_calls_to_table(discarded_gens)
        discarded_metrics_table = self._rollouts_metrics_discarded_to_table(discarded_gens)
        discarded_golden_answers_table = self._golden_answers_discarded_to_table(discarded_gens)
        discarded_samples_data_table = self._samples_data_discarded_to_table(discarded_gens)
        discarded_info_turns_table = self._info_turns_discarded_to_table(discarded_gens)
        discarded_sample_tags_table = self._sample_tags_discarded_to_table(discarded_gens)

        # Convert cancelled prompts and rollouts (reuse discarded table methods — same schema)
        cancelled_proms = cancelled_prompts or []
        cancelled_gens = cancelled_rollouts or []
        cancelled_prompts_table = self._discarded_prompts_to_table(cancelled_proms)
        cancelled_gen_table = self._discarded_generations_to_table(cancelled_gens)
        cancelled_env_resp_table = self._discarded_env_responses_to_table(cancelled_gens)
        cancelled_tool_calls_table = self._discarded_tool_calls_to_table(cancelled_gens)
        cancelled_metrics_table = self._rollouts_metrics_discarded_to_table(cancelled_gens)
        cancelled_golden_answers_table = self._golden_answers_discarded_to_table(cancelled_gens)
        cancelled_samples_data_table = self._samples_data_discarded_to_table(cancelled_gens)
        cancelled_info_turns_table = self._info_turns_discarded_to_table(cancelled_gens)
        cancelled_sample_tags_table = self._sample_tags_discarded_to_table(cancelled_gens)

        # Convert kept prompts and rollouts (empty lists if None)
        kept_proms = kept_prompts or []
        kept_gens = kept_rollouts or []
        kept_prompts_table = self._kept_prompts_to_table(kept_proms)
        kept_gen_table = self._kept_generations_to_table(kept_gens)
        kept_env_resp_table = self._kept_env_responses_to_table(kept_gens)
        kept_tool_calls_table = self._kept_tool_calls_to_table(kept_gens)
        kept_samples_data_table = self._kept_samples_data_to_table(kept_gens)
        kept_metrics_table = self._kept_rollouts_metrics_to_table(kept_gens)
        kept_golden_answers_table = self._kept_golden_answers_to_table(kept_gens)
        kept_info_turns_table = self._kept_info_turns_to_table(kept_gens)
        kept_sample_tags_table = self._kept_sample_tags_to_table(kept_gens)

        # Convert eval prompts and rollouts (empty lists if None)
        eval_rols = eval_rollouts or []
        eval_proms = eval_prompts or []
        eval_prompts_table = self._eval_prompts_to_table(eval_proms)
        eval_gen_table = self._eval_generations_to_table(eval_rols)
        eval_env_resp_table = self._eval_env_responses_to_table(eval_rols)
        eval_tool_calls_table = self._eval_tool_calls_to_table(eval_rols)
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
            pq.write_table(rollout_events_table, buf)
            zf.writestr("events_rollout.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(infra_events_table, buf)
            zf.writestr("events_infra.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(gpu_table, buf)
            zf.writestr("gpu.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cpu_table, buf)
            zf.writestr("cpu.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(vllm_table, buf)
            zf.writestr("vllm.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(thread_pool_table, buf)
            zf.writestr("thread_pools.parquet", buf.getvalue())

            # Discarded tables
            buf = io.BytesIO()
            pq.write_table(discarded_prompts_table, buf)
            zf.writestr("prompts_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_gen_table, buf)
            zf.writestr("generations_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_env_resp_table, buf)
            zf.writestr("env_responses_discarded.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(discarded_tool_calls_table, buf)
            zf.writestr("tool_calls_discarded.parquet", buf.getvalue())

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

            # Kept rollout tables
            buf = io.BytesIO()
            pq.write_table(kept_prompts_table, buf)
            zf.writestr("prompts_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_gen_table, buf)
            zf.writestr("generations_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_env_resp_table, buf)
            zf.writestr("env_responses_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_tool_calls_table, buf)
            zf.writestr("tool_calls_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_samples_data_table, buf)
            zf.writestr("samples_data_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_metrics_table, buf)
            zf.writestr("rollouts_metrics_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_golden_answers_table, buf)
            zf.writestr("golden_answers_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_info_turns_table, buf)
            zf.writestr("info_turns_kept.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(kept_sample_tags_table, buf)
            zf.writestr("sample_tags_kept.parquet", buf.getvalue())

            # Eval tables
            buf = io.BytesIO()
            pq.write_table(eval_prompts_table, buf)
            zf.writestr("prompts_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_gen_table, buf)
            zf.writestr("generations_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_env_resp_table, buf)
            zf.writestr("env_responses_eval.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(eval_tool_calls_table, buf)
            zf.writestr("tool_calls_eval.parquet", buf.getvalue())

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

            # Cancelled tables
            buf = io.BytesIO()
            pq.write_table(cancelled_prompts_table, buf)
            zf.writestr("prompts_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_gen_table, buf)
            zf.writestr("generations_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_env_resp_table, buf)
            zf.writestr("env_responses_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_tool_calls_table, buf)
            zf.writestr("tool_calls_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_samples_data_table, buf)
            zf.writestr("samples_data_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_metrics_table, buf)
            zf.writestr("rollouts_metrics_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_golden_answers_table, buf)
            zf.writestr("golden_answers_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_info_turns_table, buf)
            zf.writestr("info_turns_cancelled.parquet", buf.getvalue())

            buf = io.BytesIO()
            pq.write_table(cancelled_sample_tags_table, buf)
            zf.writestr("sample_tags_cancelled.parquet", buf.getvalue())

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
                zf.writestr("inflight.json", json.dumps(inflight_snapshot))

        self.run.save(str(dest_path), base_path=self.run.dir, policy="now")

    def _write_steps_zip_to_wandb(
        self,
        rollouts: list[GenerationRollout],
        prompts: list[Prompt],
        step_metrics: list[StepMetric],
        path: str,
        metadata: dict | None = None,
    ):
        """
        Write step data (prompts, rollouts, metrics) as parquet files inside a zip archive.

        Creates files in the zip:
        - prompts.parquet: One row per group with the initial prompt
        - generations.parquet: One row per generation (model response)
        - env_responses.parquet: One row per environment response
        - tool_calls.parquet: One row per tool call
        - samples_data.parquet: One row per sample with reward, advantage, num_generations, total_tokens, raw_string
        - rollouts_metrics.parquet: Normalized metrics table (step, sample_id, env, metric_name, value)
        - golden_answers.parquet: Golden answers table (step, sample_id, env, key, value)
        - info_turns.parquet: Per-generation info table (step, sample_id, generation_idx, env, info_key, info_value, info_type)
        - sample_tags.parquet: Per-sample string tags table (step, sample_id, env, tag_name, tag_value)
        - metrics.parquet: Per-step, per-rank metrics (step, metric, value, rank)
        - metadata.json: Archive metadata (if provided)
        """
        if self.run is None:
            return

        dest_path = Path(self.run.dir) / path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        prompts_table = self._prompts_to_table(prompts)
        gen_table = self._generations_to_table(rollouts)
        env_resp_table = self._env_responses_to_table(rollouts)
        tool_calls_table = self._tool_calls_to_table(rollouts)
        samples_data_table = self._samples_data_to_table(rollouts)
        gen_metrics_table = self._rollouts_metrics_to_table(rollouts)
        golden_answers_table = self._golden_answers_to_table(rollouts)
        info_turns_table = self._info_turns_to_table(rollouts)
        sample_tags_table = self._sample_tags_to_table(rollouts)
        turn_metrics_table = self._turn_metrics_to_table(rollouts)
        metrics_table = self._step_metrics_to_table(step_metrics)

        with zipfile.ZipFile(dest_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Write prompts table (one row per group)
            buf = io.BytesIO()
            pq.write_table(prompts_table, buf)
            zf.writestr("prompts.parquet", buf.getvalue())

            # Write generations table (one row per generation)
            buf = io.BytesIO()
            pq.write_table(gen_table, buf)
            zf.writestr("generations.parquet", buf.getvalue())

            # Write env_responses table (one row per env response)
            buf = io.BytesIO()
            pq.write_table(env_resp_table, buf)
            zf.writestr("env_responses.parquet", buf.getvalue())

            # Write tool_calls table (one row per tool call)
            buf = io.BytesIO()
            pq.write_table(tool_calls_table, buf)
            zf.writestr("tool_calls.parquet", buf.getvalue())

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

            # Write info turns table (per-generation text info: stderr, summaries, etc.)
            buf = io.BytesIO()
            pq.write_table(info_turns_table, buf)
            zf.writestr("info_turns.parquet", buf.getvalue())

            # Write sample tags table (per-sample string tags for filtering)
            buf = io.BytesIO()
            pq.write_table(sample_tags_table, buf)
            zf.writestr("sample_tags.parquet", buf.getvalue())

            # Write turn metrics table (per-turn numeric metrics)
            buf = io.BytesIO()
            pq.write_table(turn_metrics_table, buf)
            zf.writestr("turn_metrics.parquet", buf.getvalue())

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

        # Assign tail_idx to events/rollout_events/infra_events/discarded_rollouts that don't have one yet
        with self._lock:
            for e in self._events:
                if e.tail_idx == -1:
                    e.tail_idx = current_tail_idx
            for e in self._rollout_events:
                if e.tail_idx == -1:
                    e.tail_idx = current_tail_idx
            for e in self._infra_events:
                if e.tail_idx == -1:
                    e.tail_idx = current_tail_idx
            for g in self._discarded_rollouts:
                if g.tail_idx == -1:
                    g.tail_idx = current_tail_idx
            for p in self._discarded_prompts:
                if p.tail_idx == -1:
                    p.tail_idx = current_tail_idx
            for g in self._cancelled_rollouts:
                if g.tail_idx == -1:
                    g.tail_idx = current_tail_idx
            for p in self._cancelled_prompts:
                if p.tail_idx == -1:
                    p.tail_idx = current_tail_idx
            for g in self._kept_rollouts:
                if g.tail_idx == -1:
                    g.tail_idx = current_tail_idx
            for p in self._kept_prompts:
                if p.tail_idx == -1:
                    p.tail_idx = current_tail_idx
            for r in self._eval_rollouts:
                if r.tail_idx == -1:
                    r.tail_idx = current_tail_idx
            for p in self._eval_prompts:
                if p.tail_idx == -1:
                    p.tail_idx = current_tail_idx
            # Snapshot all inflight tracking dicts for tail.zip
            inflight_snapshot = {
                "timestamp": time.time(),
                "inflight_generations": list(self._inflight_generations.values()),
                "inflight_tool_executions": list(self._inflight_tool_executions.values()),
                "inflight_env_responses": list(self._inflight_env_responses.values()),
                "inflight_sandbox_ops": list(self._inflight_sandbox_ops.values()),
                "inflight_weight_syncs": list(self._inflight_weight_syncs.values()),
                "inflight_rewards": list(self._inflight_rewards.values()),
            }
            all_events = list(self._events)
            all_rollout_events = list(self._rollout_events)
            all_infra_events = list(self._infra_events)
            all_discarded_rollouts = list(self._discarded_rollouts)
            all_discarded_prompts = list(self._discarded_prompts)
            all_cancelled_rollouts = list(self._cancelled_rollouts)
            all_cancelled_prompts = list(self._cancelled_prompts)
            all_kept_rollouts = list(self._kept_rollouts)
            all_kept_prompts = list(self._kept_prompts)
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

        if self._thread_pool_metrics_logger is not None:
            new_tp = self._thread_pool_metrics_logger.get_and_clear_metrics()
            self._thread_pool_metrics.extend((m, current_tail_idx) for m in new_tp)

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
        all_thread_pool_metrics = list(self._thread_pool_metrics)
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
            block_rollout_events = [
                e for e in all_rollout_events
                if e.tail_idx >= self._block_first_tail_idx and e.tail_idx <= block_last_tail_idx
            ]
            block_infra_events = [
                e for e in all_infra_events
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
            block_thread_pool_metrics = [
                (m, idx) for m, idx in all_thread_pool_metrics
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
            block_cancelled_rollouts = [
                g for g in all_cancelled_rollouts
                if g.tail_idx >= self._block_first_tail_idx and g.tail_idx <= block_last_tail_idx
            ]
            block_cancelled_prompts = [
                p for p in all_cancelled_prompts
                if p.tail_idx >= self._block_first_tail_idx and p.tail_idx <= block_last_tail_idx
            ]
            block_kept_rollouts = [
                g for g in all_kept_rollouts
                if g.tail_idx >= self._block_first_tail_idx and g.tail_idx <= block_last_tail_idx
            ]
            block_kept_prompts = [
                p for p in all_kept_prompts
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

            if block_events or block_rollout_events or block_infra_events or block_gpu_metrics or block_cpu_metrics or block_vllm_metrics or block_discarded_rollouts or block_discarded_prompts or block_cancelled_rollouts or block_cancelled_prompts or block_kept_rollouts or block_kept_prompts or block_eval_rollouts or block_eval_prompts:
                orch_events = [e for e in block_events if e.source == "orchestrator"]
                trainer_events = [e for e in block_events if e.source == "trainer"]

                block_path = f"events/block_{self._current_block_idx}.zip"
                finalized_block_metadata = {
                    "block_idx": self._current_block_idx,
                    "min_tail_idx": self._block_first_tail_idx,
                    "max_tail_idx": block_last_tail_idx,
                }
                self._write_events_zip_to_wandb(
                    orch_events, trainer_events, block_rollout_events, block_infra_events,
                    block_gpu_metrics, block_cpu_metrics, block_vllm_metrics,
                    block_path,
                    thread_pool_metrics=block_thread_pool_metrics,
                    metadata=finalized_block_metadata,
                    discarded_rollouts=block_discarded_rollouts,
                    discarded_prompts=block_discarded_prompts,
                    cancelled_rollouts=block_cancelled_rollouts,
                    cancelled_prompts=block_cancelled_prompts,
                    kept_rollouts=block_kept_rollouts,
                    kept_prompts=block_kept_prompts,
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
                for e in self._rollout_events:
                    if e.tail_idx == -1:
                        e.tail_idx = current_tail_idx
                for e in self._infra_events:
                    if e.tail_idx == -1:
                        e.tail_idx = current_tail_idx
                for g in self._discarded_rollouts:
                    if g.tail_idx == -1:
                        g.tail_idx = current_tail_idx
                for p in self._discarded_prompts:
                    if p.tail_idx == -1:
                        p.tail_idx = current_tail_idx
                for g in self._cancelled_rollouts:
                    if g.tail_idx == -1:
                        g.tail_idx = current_tail_idx
                for p in self._cancelled_prompts:
                    if p.tail_idx == -1:
                        p.tail_idx = current_tail_idx
                for g in self._kept_rollouts:
                    if g.tail_idx == -1:
                        g.tail_idx = current_tail_idx
                for p in self._kept_prompts:
                    if p.tail_idx == -1:
                        p.tail_idx = current_tail_idx
                for r in self._eval_rollouts:
                    if r.tail_idx == -1:
                        r.tail_idx = current_tail_idx
                for p in self._eval_prompts:
                    if p.tail_idx == -1:
                        p.tail_idx = current_tail_idx
                self._events = [e for e in self._events if e.tail_idx >= new_block_first_tail_idx]
                self._rollout_events = [e for e in self._rollout_events if e.tail_idx >= new_block_first_tail_idx]
                self._infra_events = [e for e in self._infra_events if e.tail_idx >= new_block_first_tail_idx]
                self._discarded_rollouts = [g for g in self._discarded_rollouts if g.tail_idx >= new_block_first_tail_idx]
                self._discarded_prompts = [p for p in self._discarded_prompts if p.tail_idx >= new_block_first_tail_idx]
                self._cancelled_rollouts = [g for g in self._cancelled_rollouts if g.tail_idx >= new_block_first_tail_idx]
                self._cancelled_prompts = [p for p in self._cancelled_prompts if p.tail_idx >= new_block_first_tail_idx]
                self._kept_rollouts = [g for g in self._kept_rollouts if g.tail_idx >= new_block_first_tail_idx]
                self._kept_prompts = [p for p in self._kept_prompts if p.tail_idx >= new_block_first_tail_idx]
                self._eval_rollouts = [r for r in self._eval_rollouts if r.tail_idx >= new_block_first_tail_idx]
                self._eval_prompts = [p for p in self._eval_prompts if p.tail_idx >= new_block_first_tail_idx]

            # Also assign tail_idx to any new metrics added during this cycle
            self._gpu_metrics = [(m, current_tail_idx if idx == -1 else idx) for m, idx in self._gpu_metrics]
            self._gpu_metrics = [(m, idx) for m, idx in self._gpu_metrics if idx >= new_block_first_tail_idx]
            self._cpu_metrics = [(m, idx) for m, idx in self._cpu_metrics if idx >= new_block_first_tail_idx]
            self._vllm_metrics = [(m, idx) for m, idx in self._vllm_metrics if idx >= new_block_first_tail_idx]
            self._thread_pool_metrics = [(m, idx) for m, idx in self._thread_pool_metrics if idx >= new_block_first_tail_idx]
            self._log_records = [(r, idx) for r, idx in self._log_records if idx >= new_block_first_tail_idx]

            all_events = [e for e in all_events if e.tail_idx >= new_block_first_tail_idx]
            all_rollout_events = [e for e in all_rollout_events if e.tail_idx >= new_block_first_tail_idx]
            all_infra_events = [e for e in all_infra_events if e.tail_idx >= new_block_first_tail_idx]
            all_gpu_metrics = [(m, idx) for m, idx in all_gpu_metrics if idx >= new_block_first_tail_idx]
            all_cpu_metrics = [(m, idx) for m, idx in all_cpu_metrics if idx >= new_block_first_tail_idx]
            all_vllm_metrics = [(m, idx) for m, idx in all_vllm_metrics if idx >= new_block_first_tail_idx]
            all_thread_pool_metrics = [(m, idx) for m, idx in all_thread_pool_metrics if idx >= new_block_first_tail_idx]
            all_log_records = [(r, idx) for r, idx in all_log_records if idx >= new_block_first_tail_idx]
            all_discarded_rollouts = [g for g in all_discarded_rollouts if g.tail_idx >= new_block_first_tail_idx]
            all_cancelled_rollouts = [g for g in all_cancelled_rollouts if g.tail_idx >= new_block_first_tail_idx]
            all_cancelled_prompts = [p for p in all_cancelled_prompts if p.tail_idx >= new_block_first_tail_idx]
            all_kept_rollouts = [g for g in all_kept_rollouts if g.tail_idx >= new_block_first_tail_idx]
            all_kept_prompts = [p for p in all_kept_prompts if p.tail_idx >= new_block_first_tail_idx]
            all_eval_rollouts = [r for r in all_eval_rollouts if r.tail_idx >= new_block_first_tail_idx]
            all_eval_prompts = [p for p in all_eval_prompts if p.tail_idx >= new_block_first_tail_idx]

            # Set the new block's first tail_idx
            self._block_first_tail_idx = new_block_first_tail_idx

        # Filter for current block BY TAIL_IDX (ensures complete tails)
        current_block_events = [e for e in all_events if e.tail_idx >= self._block_first_tail_idx]
        current_block_rollout_events = [e for e in all_rollout_events if e.tail_idx >= self._block_first_tail_idx]
        current_block_infra_events = [e for e in all_infra_events if e.tail_idx >= self._block_first_tail_idx]
        current_block_gpu = [(m, idx) for m, idx in all_gpu_metrics if idx >= self._block_first_tail_idx]
        current_block_cpu = [(m, idx) for m, idx in all_cpu_metrics if idx >= self._block_first_tail_idx]
        current_block_vllm = [(m, idx) for m, idx in all_vllm_metrics if idx >= self._block_first_tail_idx]
        current_block_thread_pools = [(m, idx) for m, idx in all_thread_pool_metrics if idx >= self._block_first_tail_idx]
        current_block_discarded_rollouts = [g for g in all_discarded_rollouts if g.tail_idx >= self._block_first_tail_idx]
        current_block_discarded_prompts = [p for p in all_discarded_prompts if p.tail_idx >= self._block_first_tail_idx]
        current_block_cancelled_rollouts = [g for g in all_cancelled_rollouts if g.tail_idx >= self._block_first_tail_idx]
        current_block_cancelled_prompts = [p for p in all_cancelled_prompts if p.tail_idx >= self._block_first_tail_idx]
        current_block_kept_rollouts = [g for g in all_kept_rollouts if g.tail_idx >= self._block_first_tail_idx]
        current_block_kept_prompts = [p for p in all_kept_prompts if p.tail_idx >= self._block_first_tail_idx]
        current_block_eval_rollouts = [r for r in all_eval_rollouts if r.tail_idx >= self._block_first_tail_idx]
        current_block_eval_prompts = [p for p in all_eval_prompts if p.tail_idx >= self._block_first_tail_idx]
        current_block_log_records = [(r, idx) for r, idx in all_log_records if idx >= self._block_first_tail_idx]

        # Filter for tail BY TAIL_IDX (ensures complete tails)
        tail_events = [e for e in all_events if e.tail_idx >= tail_min_idx_for_window]
        tail_rollout_events = [e for e in all_rollout_events if e.tail_idx >= tail_min_idx_for_window]
        tail_infra_events = [e for e in all_infra_events if e.tail_idx >= tail_min_idx_for_window]
        tail_gpu = [(m, idx) for m, idx in all_gpu_metrics if idx >= tail_min_idx_for_window]
        tail_cpu = [(m, idx) for m, idx in all_cpu_metrics if idx >= tail_min_idx_for_window]
        tail_vllm = [(m, idx) for m, idx in all_vllm_metrics if idx >= tail_min_idx_for_window]
        tail_thread_pools = [(m, idx) for m, idx in all_thread_pool_metrics if idx >= tail_min_idx_for_window]
        tail_discarded_rollouts = [g for g in all_discarded_rollouts if g.tail_idx >= tail_min_idx_for_window]
        tail_discarded_prompts = [p for p in all_discarded_prompts if p.tail_idx >= tail_min_idx_for_window]
        tail_cancelled_rollouts = [g for g in all_cancelled_rollouts if g.tail_idx >= tail_min_idx_for_window]
        tail_cancelled_prompts = [p for p in all_cancelled_prompts if p.tail_idx >= tail_min_idx_for_window]
        tail_kept_rollouts = [g for g in all_kept_rollouts if g.tail_idx >= tail_min_idx_for_window]
        tail_kept_prompts = [p for p in all_kept_prompts if p.tail_idx >= tail_min_idx_for_window]
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
            [e.timestamp for e in tail_rollout_events] +
            [e.timestamp for e in tail_infra_events] +
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
            tail_orch_events, tail_trainer_events, tail_rollout_events, tail_infra_events,
            tail_gpu, tail_cpu, tail_vllm,
            "events/tail.zip",
            metadata=tail_metadata,
            discarded_rollouts=tail_discarded_rollouts,
            discarded_prompts=tail_discarded_prompts,
            cancelled_rollouts=tail_cancelled_rollouts,
            cancelled_prompts=tail_cancelled_prompts,
            kept_rollouts=tail_kept_rollouts,
            kept_prompts=tail_kept_prompts,
            eval_rollouts=tail_eval_rollouts,
            eval_prompts=tail_eval_prompts,
            logs=tail_log_records,
            inflight_snapshot=inflight_snapshot,
            thread_pool_metrics=tail_thread_pools,
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
            block_orch_events, block_trainer_events, current_block_rollout_events, current_block_infra_events,
            current_block_gpu, current_block_cpu, current_block_vllm,
            "events/block_live.zip",
            metadata=block_live_metadata,
            discarded_rollouts=current_block_discarded_rollouts,
            discarded_prompts=current_block_discarded_prompts,
            cancelled_rollouts=current_block_cancelled_rollouts,
            cancelled_prompts=current_block_cancelled_prompts,
            kept_rollouts=current_block_kept_rollouts,
            kept_prompts=current_block_kept_prompts,
            eval_rollouts=current_block_eval_rollouts,
            eval_prompts=current_block_eval_prompts,
            logs=current_block_log_records,
            thread_pool_metrics=current_block_thread_pools,
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
            "events/tail_rollout_events_count": len(tail_rollout_events),
            "events/tail_infra_events_count": len(tail_infra_events),
            "events/tail_gpu_count": len(tail_gpu),
            "events/tail_cpu_count": len(tail_cpu),
            "events/tail_vllm_count": len(tail_vllm),
            "events/tail_discarded_rollouts_count": len(tail_discarded_rollouts),
            "events/tail_cancelled_rollouts_count": len(tail_cancelled_rollouts),
            # Events counts in block_live
            "events/block_live_orchestrator_count": len(block_orch_events),
            "events/block_live_trainer_count": len(block_trainer_events),
            "events/block_live_rollout_events_count": len(current_block_rollout_events),
            "events/block_live_infra_events_count": len(current_block_infra_events),
            "events/block_live_gpu_count": len(current_block_gpu),
            "events/block_live_cpu_count": len(current_block_cpu),
            "events/block_live_vllm_count": len(current_block_vllm),
            "events/block_live_discarded_rollouts_count": len(current_block_discarded_rollouts),
            "events/block_live_cancelled_rollouts_count": len(current_block_cancelled_rollouts),
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
        _log.debug(f"Uploaded tail {current_tail_idx}: events={len(tail_orch_events)}+{len(tail_trainer_events)}+{len(tail_rollout_events)}+{len(tail_infra_events)}, metrics={len(tail_gpu)}+{len(tail_cpu)}+{len(tail_vllm)}, discarded={len(tail_discarded_rollouts)}, cancelled={len(tail_cancelled_rollouts)}")

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

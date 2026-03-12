"""Pydantic schema for Telescope configuration.

Flat layout — every field name is globally unique and self-descriptive.
Using ``extra="forbid"`` so typos in YAML keys are caught at load time.

Default values live in ``configs/defaults/default_train.yaml`` (the single source of truth).
This schema only defines types and validators.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models for array-of-table entries
# ---------------------------------------------------------------------------

class EnvironmentEntry(BaseModel, extra="forbid"):
    name: str
    weight: float = Field(default=1.0, gt=0)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    reward_min: float | None = None
    reward_max: float | None = None


class EvalEntry(BaseModel, extra="forbid"):
    name: str
    eval_every: int = Field(default=10, ge=1)
    pass_k: dict[str, Any] = Field(default_factory=dict)
    num_samples: int = -1
    separate_eval_samples: bool = False
    kwargs: dict[str, Any] = Field(default_factory=dict)
    # Sampling overrides (None = inherit from base config)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    def get_sampling_overrides(self) -> dict[str, Any]:
        """Return only the non-None sampling overrides as a dict."""
        d: dict[str, Any] = {}
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_p is not None:
            d["top_p"] = self.top_p
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        return d


# ---------------------------------------------------------------------------
# Root config — flat, every field is self-descriptive
# ---------------------------------------------------------------------------

class TelescopeConfig(BaseModel, extra="forbid"):

    _custom_config: dict[str, Any] = PrivateAttr(default_factory=dict)

    def get_custom_config(self) -> dict[str, Any]:
        """Return the user-specified config (run YAML + CLI overrides)."""
        return dict(self._custom_config)

    # General
    debug: bool

    # Model
    model: str
    model_dtype: Literal["float32", "float16", "bfloat16"]
    mixed_precision_dtype: Literal["float32", "float16", "bfloat16"] | None

    # Environments (must be provided in run config)
    environments: list[EnvironmentEntry]

    @field_validator("environments")
    @classmethod
    def _environments_not_empty(cls, v: list[EnvironmentEntry]) -> list[EnvironmentEntry]:
        if not v:
            raise ValueError(
                "No environments configured. "
                "You must specify at least one environment in your config, e.g.:\n"
                "  environments:\n"
                "    - name: \"countdown\"\n"
                "      weight: 1.0\n"
                "      reward_min: 0.0\n"
                "      reward_max: 2.0"
            )
        return v

    # Ray
    ray_address: str
    ray_auto_start_local: bool
    ray_namespace: str
    ray_log_to_driver: bool
    ray_runtime_env: dict[str, Any] | None
    ray_disable_runtime_env_hook: bool
    ray_pin_py_executable: bool
    ray_propagate_active_venv: bool
    ray_propagate_run_dir: bool
    ray_broadcast_init_timeout_s: int
    ray_broadcast_prefer_loopback_if_single_node: bool
    ray_shutdown_on_exit: bool
    ray_inference_cpus_per_worker: float
    ray_trainer_cpus_per_worker: float
    ray_inference_placement_strategy: Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"]
    ray_trainer_placement_strategy: Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"]
    ray_placement_timeout_s: int

    # Workers
    inference_num_workers: int = Field(ge=1)
    inference_tensor_parallel_size: int = Field(ge=1)
    trainer_num_workers: int = Field(ge=1)

    # Orchestrator
    max_concurrent_prompts_per_server: int = Field(ge=1)
    prompts_batch_size_for_trainer: int = Field(ge=1)
    number_of_steps: int = Field(ge=1)
    max_async_rollout: int = Field(ge=0)
    discard_group_zero_advantage: bool
    enable_prompt_prefetch: bool
    prompt_prefetch_buffer_size: int
    enable_individual_sample_lanes: bool
    max_off_policy_steps: int

    # Trainer
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(ge=0)
    lr_scheduler: Literal["none", "constant", "linear", "cosine"]
    warmup_steps: int = Field(ge=0)
    min_lr_ratio: float = Field(ge=0, le=1)
    train_backend: Literal["fsdp", "megatron"]

    # Megatron
    megatron_tensor_parallel_size: int = Field(ge=1)
    megatron_pipeline_parallel_size: int = Field(ge=1)
    megatron_context_parallel_size: int = Field(ge=1)
    megatron_expert_parallel_size: int = Field(ge=1)
    megatron_global_batch_size: int | None
    megatron_disable_unified_memory_jit: bool
    megatron_optimizer_cpu_offload: bool
    megatron_optimizer_offload_fraction: float = Field(ge=0, le=1)
    megatron_overlap_cpu_optimizer_d2h_h2d: bool
    megatron_use_precision_aware_optimizer: bool
    megatron_main_grads_dtype: str
    megatron_main_params_dtype: str
    megatron_exp_avg_dtype: str
    megatron_exp_avg_sq_dtype: str
    megatron_grad_reduce_in_fp32: bool
    megatron_gradient_checkpointing: bool
    megatron_sequence_parallel: bool
    megatron_use_distributed_optimizer: bool
    megatron_overlap_grad_reduce: bool
    megatron_use_transformer_engine: bool
    megatron_fp8: bool

    # Algorithm
    algorithm: Literal["grpo", "rloo", "reinforce_pp", "dr_grpo", "cispo", "gspo", "sapo"]
    number_of_minibatches: int = Field(ge=1)
    ppo_clip_ref_logprobs: Literal["rollout", "batch"]
    clip_low: float = Field(ge=0, le=1)
    clip_high: float = Field(ge=0)
    sapo_tau_pos: float = Field(gt=0)
    sapo_tau_neg: float = Field(gt=0)
    dr_grpo_loss_agg_mode: Literal["token_mean", "token_sum_norm"]
    advantage_norm: Literal["group", "batch"]
    use_ppo_clip: bool
    use_tis: bool
    tis_cap: float = Field(ge=1.0)
    tis_logprob_clamp: float = Field(gt=0)
    entropy_chunk_size: int

    # Weight sync
    weight_broadcast_mode: Literal["flattened_bucket", "per_tensor"]
    weight_broadcast_bucket_mb: int = Field(ge=1)
    weight_broadcast_cpu_staging: bool
    weight_broadcast_pin_memory: bool
    weight_broadcast_free_grad_buffers: bool

    # Sequence packing
    seq_len: int = Field(ge=1)
    pad_to_multiple_of: int = Field(ge=1)

    # Inference server
    inference_host: str
    inference_base_port: int = Field(ge=1, le=65535)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int = Field(ge=1)
    vllm_scheduling_policy: Literal["priority", "fcfs"]
    enable_thinking: bool
    chat_template: str | None = None

    # Rollout / sampling
    group_size: int = Field(ge=1)
    temperature: float = Field(ge=0)
    top_p: float | None = Field(gt=0, le=1)
    max_tokens: int = Field(ge=1)
    interleaved_rollouts: bool

    def get_sampling_params(self) -> dict[str, Any]:
        """Build sampling params dict for vLLM API calls."""
        d: dict[str, Any] = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return d

    # Checkpoint
    checkpoint_every: int | bool
    checkpoint_save_training_state: bool
    resume_from_checkpoint: bool | int
    checkpoint_dir: str | None
    checkpoint_keep_last: int | None = Field(ge=1)
    checkpoint_keep_every: int | None = Field(ge=1)

    # Logging
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str
    wandb_tags: list[str]
    wandb_upload_code: bool
    wandb_code_max_file_size_mb: float = Field(gt=0)
    wandb_code_exclude_patterns: list[str]
    system_metrics_collection_interval_seconds: float
    torch_memory_sample_interval_seconds: float
    event_tail_window_seconds: int
    event_block_duration_seconds: int
    event_upload_interval_seconds: int
    metrics_logger_interval_seconds: float
    ray_torch_memory_drain_interval_seconds: float
    rollout_block_size: int
    track_gpu_events: bool

    # Evals
    eval_before_training: bool
    eval_after_training: bool
    eval_num_servers: int = Field(ge=0)
    eval_start_end_use_all_servers: bool
    evals: list[EvalEntry] = Field(default_factory=list)

    # vLLM tracing
    enable_vllm_tracing: bool
    otlp_receiver_port: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def _check_tis_ppo_clip_conflict(self) -> TelescopeConfig:
        if self.use_tis and self.use_ppo_clip and self.ppo_clip_ref_logprobs == "rollout":
            raise ValueError(
                "use_tis=true + use_ppo_clip=true + ppo_clip_ref_logprobs='rollout' "
                "double-counts the IS correction (both use π_current/π_rollout). "
                "Set ppo_clip_ref_logprobs='batch' so PPO uses π_current/π_old "
                "while TIS uses π_current/π_rollout."
            )
        return self

    @model_validator(mode="after")
    def _check_ppo_clip_algo_conflict(self) -> TelescopeConfig:
        if self.use_ppo_clip and self.algorithm in ("cispo", "gspo", "sapo"):
            raise ValueError(
                f"use_ppo_clip=true is incompatible with algorithm={self.algorithm!r}. "
                f"{self.algorithm!r} has its own clipping mechanism."
            )
        return self

    @model_validator(mode="after")
    def _check_advantage_norm_algo_conflict(self) -> TelescopeConfig:
        if self.advantage_norm == "batch" and self.algorithm in ("rloo", "dr_grpo"):
            raise ValueError(
                f"advantage_norm='batch' is incompatible with algorithm={self.algorithm!r}. "
                f"{self.algorithm!r} has fixed per-group advantage semantics "
                f"(batch-level normalization would be silently ignored)."
            )
        if self.advantage_norm == "group" and self.algorithm == "reinforce_pp":
            raise ValueError(
                "advantage_norm='group' is incompatible with algorithm='reinforce_pp'. "
                "REINFORCE++ requires batch-level normalization (arXiv:2501.03262). "
                "Set advantage_norm='batch'."
            )
        return self

    @model_validator(mode="after")
    def _check_worker_tp_divisibility(self) -> TelescopeConfig:
        if self.inference_num_workers % self.inference_tensor_parallel_size != 0:
            raise ValueError(
                f"inference_num_workers ({self.inference_num_workers}) must be divisible by "
                f"inference_tensor_parallel_size ({self.inference_tensor_parallel_size})"
            )
        return self

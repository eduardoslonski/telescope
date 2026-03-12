"""Pydantic schema for standalone eval configuration.

Flat layout with ``extra="forbid"`` so typos are caught at load time.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class CheckpointEntry(BaseModel, extra="forbid"):
    """A single checkpoint to evaluate."""
    path: str       # HF-format checkpoint directory
    step: int = Field(ge=0)  # training step this checkpoint corresponds to


class StandaloneEvalEntry(BaseModel, extra="forbid"):
    """An eval to run in standalone (post-training) mode.

    Same fields as the training ``EvalEntry`` but without ``eval_every``
    (standalone eval runs every eval on every checkpoint).
    """
    name: str
    pass_k: dict[str, Any] = Field(default_factory=dict)
    num_samples: int = -1
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


class EvalStandaloneConfig(BaseModel, extra="forbid"):
    """Configuration for standalone (post-training) evaluation."""

    # Required fields
    wandb_run_path: str                          # "entity/project/run_id"
    evals: list[StandaloneEvalEntry]

    # Checkpoint sources (at least one required)
    checkpoints: list[CheckpointEntry] = Field(default_factory=list)  # explicit list
    checkpoint_dir: str | None = None            # auto-discover step_N/ dirs

    # When True, checkpoint_dir is expected to contain already-converted HF
    # checkpoints (step_N/ with config.json).  When False (default), it
    # contains native training checkpoints (step_N/ with meta.json) that are
    # converted to HF format on the fly before each eval.
    hf_weights: bool = False

    # Inference settings (subset of TelescopeConfig)
    inference_num_workers: int = Field(default=1, ge=1)
    inference_tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    max_model_len: int = Field(default=4000, ge=1)
    max_concurrent_samples_per_server: int = Field(default=256, ge=1)
    vllm_scheduling_policy: Literal["priority", "fcfs"] = "priority"
    enable_thinking: bool = False
    chat_template: str | None = None

    # Sampling defaults
    temperature: float = Field(default=1.0, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int = Field(default=3700, ge=1)

    # Ray settings (minimal subset)
    ray_address: str = "auto"
    ray_auto_start_local: bool = True
    ray_namespace: str = "telescope-eval"
    ray_log_to_driver: bool = True
    ray_disable_runtime_env_hook: bool = True
    ray_pin_py_executable: bool = True
    ray_propagate_active_venv: bool = True
    ray_propagate_run_dir: bool = True
    ray_inference_cpus_per_worker: float = 4.0
    ray_inference_placement_strategy: Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"] = "PACK"
    ray_placement_timeout_s: int = 900
    ray_shutdown_on_exit: bool = False
    ray_runtime_env: dict[str, Any] | None = None

    # Misc
    enable_prompt_prefetch: bool = True
    inference_host: str = "0.0.0.0"
    inference_base_port: int = Field(default=8100, ge=1, le=65535)

    @model_validator(mode="after")
    def _check_checkpoint_source(self) -> EvalStandaloneConfig:
        if not self.checkpoints and not self.checkpoint_dir:
            raise ValueError("At least one of 'checkpoints' or 'checkpoint_dir' must be set")
        return self

    def get_sampling_params(self) -> dict[str, Any]:
        """Build sampling params dict for vLLM API calls."""
        d: dict[str, Any] = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return d

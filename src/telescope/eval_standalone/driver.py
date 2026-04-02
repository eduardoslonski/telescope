"""Standalone eval driver — evaluate checkpoints and log to an existing wandb run.

Supports both pre-converted HF checkpoints (``hf_weights=True``) and native
training checkpoints that are converted on the fly (``hf_weights=False``,
the default).

Main flow:
1. Parse eval config and resolve checkpoints
2. Load tokenizer from first checkpoint
3. Init Ray cluster
4. Attach to wandb run (resume="allow")
5. Create EvalRunner, load eval environments
6. For each checkpoint (sorted by step):
   a. Convert to HF format if needed (hf_weights=False)
   b. Install compat config with model=checkpoint.path
   c. Start RayInferenceGroup (vLLM loads checkpoint)
   d. Run all evals via EvalRunner.run_evals()
   e. Collect eval results
   f. Stop RayInferenceGroup
   g. Clean up converted checkpoint if applicable
7. Upload all results as a single zip to evals_after_training/
8. Cleanup
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import wandb
import zstandard as zstd
from transformers import AutoTokenizer

import telescope.utils.config as config_module
from telescope import table_schema_versions
from telescope.eval_standalone.config_schema import CheckpointEntry, EvalStandaloneConfig
from telescope.orchestrator.eval_runner import (
    EvalConfig,
    EvalRunner,
    PassKConfig,
    PassKEntry,
)
from telescope.orchestrator.loggers.event_logger import (
    EvalPrompt,
    EvalGenerationRollout,
    GenerationRecord,
    EnvResponseRecord,
    PROMPTS_EVAL_SCHEMA,
    GENERATIONS_EVAL_SCHEMA,
    ENV_RESPONSES_EVAL_SCHEMA,
    TOOL_CALLS_EVAL_SCHEMA,
    SAMPLES_DATA_EVAL_SCHEMA,
    ROLLOUTS_METRICS_EVAL_SCHEMA,
    GOLDEN_ANSWERS_EVAL_SCHEMA,
    INFO_TURNS_EVAL_SCHEMA,
    SAMPLE_TAGS_EVAL_SCHEMA,
)
from telescope.utils.config_schema import TelescopeConfig
from telescope.utils.ray_runtime.runtime import (
    RayInferenceGroup,
    init_ray_cluster,
)
from telescope.utils.tlog import get_logger, setup_logging

_log = get_logger("eval-standalone")

_STEP_RE = re.compile(r"^step_(\d+)$")


# ---------------------------------------------------------------------------
# Checkpoint auto-discovery
# ---------------------------------------------------------------------------

def _discover_checkpoints(checkpoint_dir: str, *, hf_weights: bool) -> list[CheckpointEntry]:
    """Scan a directory for checkpoints.

    When *hf_weights* is ``True``, looks for HF checkpoints (step_N/ with
    ``config.json``).  When ``False``, looks for native training checkpoints
    (step_N/ with ``meta.json``).
    """
    marker = "config.json" if hf_weights else "meta.json"
    label = "HF" if hf_weights else "native"
    entries: list[CheckpointEntry] = []
    abs_checkpoint_dir = os.path.abspath(checkpoint_dir)
    for name in os.listdir(abs_checkpoint_dir):
        m = _STEP_RE.match(name)
        if not m:
            continue
        path = os.path.join(abs_checkpoint_dir, name)
        if not os.path.isdir(path):
            continue
        if not os.path.isfile(os.path.join(path, marker)):
            _log.warning(f"Skipping {name}: no {marker} (not a {label} checkpoint?)")
            continue
        entries.append(CheckpointEntry(path=path, step=int(m.group(1))))
    entries.sort(key=lambda e: e.step)
    return entries


def _resolve_checkpoints(eval_cfg: EvalStandaloneConfig) -> list[CheckpointEntry]:
    """Merge explicit checkpoints with auto-discovered ones. Explicit wins on same step."""
    by_step: dict[int, CheckpointEntry] = {}

    if eval_cfg.checkpoint_dir:
        for entry in _discover_checkpoints(
            eval_cfg.checkpoint_dir, hf_weights=eval_cfg.hf_weights,
        ):
            by_step[entry.step] = entry

    # Explicit checkpoints override auto-discovered ones (resolve to absolute paths)
    for entry in eval_cfg.checkpoints:
        resolved = CheckpointEntry(path=os.path.abspath(entry.path), step=entry.step)
        by_step[resolved.step] = resolved

    resolved = sorted(by_step.values(), key=lambda e: e.step)
    if not resolved:
        marker = "config.json" if eval_cfg.hf_weights else "meta.json"
        raise ValueError(
            f"No checkpoints found. Check that checkpoint_dir contains "
            f"step_N/ directories with {marker}."
        )
    return resolved


# ---------------------------------------------------------------------------
# On-the-fly checkpoint conversion
# ---------------------------------------------------------------------------

def _convert_checkpoint(ckpt_path: str, output_dir: Path) -> str:
    """Convert a native training checkpoint to HF format.

    Returns the path to the converted HF checkpoint.  If the output already
    exists (e.g. from a previous interrupted run), it is reused.
    """
    from telescope.utils.checkpoint_converter import convert_single

    if (output_dir / "config.json").is_file():
        _log.info(f"Reusing existing converted checkpoint at {output_dir}")
        return str(output_dir)

    _log.info(f"Converting native checkpoint {ckpt_path} -> {output_dir} ...")
    try:
        convert_single(Path(ckpt_path), output_dir)
    except Exception:
        # Clean up partial output so it won't be reused on retry
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        raise
    _log.info(f"Conversion complete: {output_dir}")
    return str(output_dir)


def _cleanup_converted(path: Path) -> None:
    """Remove a converted checkpoint directory."""
    if path.is_dir():
        _log.info(f"Cleaning up converted checkpoint: {path}")
        shutil.rmtree(path)


# ---------------------------------------------------------------------------
# Config bridge
# ---------------------------------------------------------------------------

def _install_compat_config(
    eval_cfg: EvalStandaloneConfig,
    model_path: str,
) -> None:
    """Build a synthetic TelescopeConfig and install it as the global singleton.

    Existing modules (EvalRunner, generate.py, inference/server.py) all read
    from ``telescope.utils.config``.  Instead of refactoring them, we build a
    compat config that exposes the fields they need.
    """
    compat = TelescopeConfig(
        model=model_path,
        # Inference
        inference_num_workers=eval_cfg.inference_num_workers,
        inference_tensor_parallel_size=eval_cfg.inference_tensor_parallel_size,
        gpu_memory_utilization=eval_cfg.gpu_memory_utilization,
        max_model_len=eval_cfg.max_model_len,
        max_concurrent_prompts_per_server=eval_cfg.max_concurrent_samples_per_server,
        vllm_scheduling_policy=eval_cfg.vllm_scheduling_policy,
        enable_thinking=eval_cfg.enable_thinking,
        chat_template=eval_cfg.chat_template,
        # Sampling
        temperature=eval_cfg.temperature,
        top_p=eval_cfg.top_p,
        max_tokens=eval_cfg.max_tokens,
        # Ray
        ray_address=eval_cfg.ray_address,
        ray_auto_start_local=eval_cfg.ray_auto_start_local,
        ray_namespace=eval_cfg.ray_namespace,
        ray_log_to_driver=eval_cfg.ray_log_to_driver,
        ray_disable_runtime_env_hook=eval_cfg.ray_disable_runtime_env_hook,
        ray_pin_py_executable=eval_cfg.ray_pin_py_executable,
        ray_propagate_active_venv=eval_cfg.ray_propagate_active_venv,
        ray_propagate_run_dir=eval_cfg.ray_propagate_run_dir,
        ray_inference_cpus_per_worker=eval_cfg.ray_inference_cpus_per_worker,
        ray_inference_placement_strategy=eval_cfg.ray_inference_placement_strategy,
        ray_placement_timeout_s=eval_cfg.ray_placement_timeout_s,
        ray_shutdown_on_exit=eval_cfg.ray_shutdown_on_exit,
        ray_runtime_env=eval_cfg.ray_runtime_env,
        # Misc
        enable_prompt_prefetch=eval_cfg.enable_prompt_prefetch,
        enable_vllm_tracing=False,
        inference_host=eval_cfg.inference_host,
        inference_base_port=eval_cfg.inference_base_port,
        # Safe defaults for fields the compat config needs but standalone eval doesn't use
        trainer_num_workers=1,
        use_wandb=True,
        eval_num_servers=0,
    )
    config_module._cfg = compat


# ---------------------------------------------------------------------------
# Eval config parsing (replicates orchestrator._parse_eval_configs)
# ---------------------------------------------------------------------------

def _parse_eval_configs(eval_cfg: EvalStandaloneConfig) -> list[EvalConfig]:
    """Parse ``StandaloneEvalEntry`` list into strongly-typed ``EvalConfig`` objects."""
    base_sampling = eval_cfg.get_sampling_params()
    configs: list[EvalConfig] = []

    for entry in eval_cfg.evals:
        raw_pass_k = entry.pass_k
        if isinstance(raw_pass_k, dict):
            raw_at_k = raw_pass_k.get("at_k", {})
            raw_pow_k = raw_pass_k.get("pow_k", {})
            pass_k_cfg = PassKConfig(
                at_k=PassKEntry(
                    metrics=list(raw_at_k.get("metrics", [])),
                    k=list(raw_at_k.get("k", [1])),
                ),
                pow_k=PassKEntry(
                    metrics=list(raw_pow_k.get("metrics", [])),
                    k=list(raw_pow_k.get("k", [])),
                ),
            )
        else:
            pass_k_cfg = PassKConfig(at_k=PassKEntry(k=list(raw_pass_k)))

        eval_sampling = dict(base_sampling)
        eval_sampling.update(entry.get_sampling_overrides())

        configs.append(EvalConfig(
            name=entry.name,
            eval_every=1,  # standalone eval runs every eval on every checkpoint
            pass_k=pass_k_cfg,
            num_samples=entry.num_samples,
            kwargs=dict(entry.kwargs),
            sampling_params=eval_sampling,
        ))

    return configs


# ---------------------------------------------------------------------------
# Result collection (builds EvalPrompt/EvalGenerationRollout objects)
# ---------------------------------------------------------------------------

_zstd_compressor = zstd.ZstdCompressor(level=3)


def _zstd_compress(text: str) -> bytes:
    return _zstd_compressor.compress(text.encode("utf-8"))


def _collect_eval_objects(
    results: list[dict[str, Any]],
    step: int,
    model_step: int,
    tokenizer: Any | None = None,
) -> tuple[list[EvalPrompt], list[EvalGenerationRollout]]:
    """Convert eval results into EvalPrompt/EvalGenerationRollout objects.

    Returns (eval_prompts, eval_rollouts).
    """
    eval_prompts: list[EvalPrompt] = []
    eval_rollouts: list[EvalGenerationRollout] = []
    logged_prompt_keys: set[tuple[str, int]] = set()

    for result in results:
        if not isinstance(result, dict):
            continue
        eval_name = result.get("eval_name", "unknown")
        avg_metrics = result.get("avg_metrics", {})
        num_samples = result.get("num_samples", 0)
        num_completions = result.get("num_completions", 0)

        _log.info(
            f"Eval {eval_name} (step={step}): "
            f"{num_samples} samples, {num_completions} completions, "
            f"metrics={avg_metrics}",
        )

        # Collect per-sample EvalPrompt and EvalGenerationRollout objects
        sample_results = result.get("sample_results", [])
        for sr in sample_results:
            # Deduplicate prompts (same eval_name + sample_idx across pass@k completions)
            prompt_key = (eval_name, sr.sample_idx)
            if prompt_key not in logged_prompt_keys:
                logged_prompt_keys.add(prompt_key)
                tokens_prompt = 0
                tokens_system_prompt = 0
                if tokenizer is not None:
                    if sr.prompt_text:
                        tokens_prompt = len(tokenizer.encode(sr.prompt_text))
                    if sr.system_prompt:
                        tokens_system_prompt = len(tokenizer.encode(sr.system_prompt))

                eval_prompts.append(EvalPrompt(
                    step=step,
                    eval_name=eval_name,
                    model_step=model_step,
                    sample_idx=sr.sample_idx,
                    sample_id=sr.sample_id,
                    env=sr.env_name or eval_name,
                    prompt=sr.prompt_text,
                    tokens_prompt=tokens_prompt,
                    system_prompt=sr.system_prompt,
                    tokens_system_prompt=tokens_system_prompt,
                    tail_idx=0,
                ))

            # Split turns into GenerationRecord and EnvResponseRecord lists
            generations: list[GenerationRecord] = []
            env_responses: list[EnvResponseRecord] = []
            generation_idx = 0
            last_generation_idx = 0

            for t in sr.turns:
                turn_type = t.get("turn_type", "model")
                if turn_type == "model":
                    generations.append(GenerationRecord(
                        generation_idx=generation_idx,
                        content=t.get("content", ""),
                        tokens=t.get("tokens", 0),
                        stop_reason=t.get("stop_reason", ""),
                    ))
                    last_generation_idx = generation_idx
                    generation_idx += 1
                else:
                    env_responses.append(EnvResponseRecord(
                        generation_idx=last_generation_idx,
                        content=t.get("content", ""),
                        turn_type=turn_type,
                        tokens=t.get("tokens", 0),
                        response_time=t.get("environment_response_time", 0.0),
                    ))

            # Determine overall stop_reason from the last generation
            stop_reason = generations[-1].stop_reason if generations else ""

            eval_rollouts.append(EvalGenerationRollout(
                step=step,
                eval_name=eval_name,
                model_step=model_step,
                sample_idx=sr.sample_idx,
                sample_id=sr.sample_id,
                completion_idx=sr.completion_idx,
                env=sr.env_name or eval_name,
                generations=generations,
                env_responses=env_responses,
                sample_metrics=sr.metrics,
                golden_answers=sr.golden_answers,
                info_turns=sr.info_turns,
                sample_tags=sr.sample_tags,
                compute_eval_metrics_time=sr.compute_eval_metrics_time,
                stop_reason=stop_reason,
                tail_idx=0,
            ))

    return eval_prompts, eval_rollouts


# ---------------------------------------------------------------------------
# Parquet table builders (mirror EventLogger methods as standalone functions)
# ---------------------------------------------------------------------------

def _eval_prompts_to_table(prompts: list[EvalPrompt]) -> pa.Table:
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


def _eval_generations_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in GENERATIONS_EVAL_SCHEMA})
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


def _eval_env_responses_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in ENV_RESPONSES_EVAL_SCHEMA})
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


def _eval_tool_calls_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in TOOL_CALLS_EVAL_SCHEMA})
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


def _samples_data_eval_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in SAMPLES_DATA_EVAL_SCHEMA})
    data = {
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
    }
    return pa.table(data, schema=SAMPLES_DATA_EVAL_SCHEMA)


def _rollouts_metrics_eval_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in ROLLOUTS_METRICS_EVAL_SCHEMA})
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
    data = {
        "step": steps, "eval_name": eval_names,
        "sample_idx": sample_idxs, "sample_id": sample_ids,
        "completion_idx": comp_idxs,
        "env": envs, "metric_name": metric_names,
        "value": values, "tail_idx": tail_idxs,
    }
    return pa.table(data, schema=ROLLOUTS_METRICS_EVAL_SCHEMA)


def _golden_answers_eval_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in GOLDEN_ANSWERS_EVAL_SCHEMA})
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
    data = {
        "step": steps, "eval_name": eval_names,
        "sample_idx": sample_idxs, "sample_id": sample_ids,
        "completion_idx": comp_idxs,
        "env": envs, "key": keys, "value": values, "tail_idx": tail_idxs,
    }
    return pa.table(data, schema=GOLDEN_ANSWERS_EVAL_SCHEMA)


def _info_turns_eval_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in INFO_TURNS_EVAL_SCHEMA})
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
            gen_idxs.append(info.get("generation_idx", 0))
            tc_idxs.append(info.get("tool_call_idx", -1))
            envs.append(r.env)
            info_keys.append(info.get("info_key", ""))
            info_values.append(info.get("info_value", ""))
            info_types.append(info.get("info_type", "text"))
            tail_idxs.append(r.tail_idx)
    data = {
        "step": steps, "eval_name": eval_names,
        "sample_idx": sample_idxs, "sample_id": sample_ids,
        "completion_idx": comp_idxs,
        "agent_id": agent_ids, "generation_idx": gen_idxs, "tool_call_idx": tc_idxs,
        "env": envs, "info_key": info_keys, "info_value": info_values,
        "info_type": info_types, "tail_idx": tail_idxs,
    }
    return pa.table(data, schema=INFO_TURNS_EVAL_SCHEMA)


def _sample_tags_eval_to_table(rollouts: list[EvalGenerationRollout]) -> pa.Table:
    if not rollouts:
        return pa.table({f.name: pa.array([], type=f.type) for f in SAMPLE_TAGS_EVAL_SCHEMA})
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
    data = {
        "step": steps, "eval_name": eval_names,
        "sample_idx": sample_idxs, "sample_id": sample_ids,
        "completion_idx": comp_idxs,
        "env": envs, "tag_name": tag_names,
        "tag_value": tag_values, "tail_idx": tail_idxs,
    }
    return pa.table(data, schema=SAMPLE_TAGS_EVAL_SCHEMA)


# ---------------------------------------------------------------------------
# Zip upload
# ---------------------------------------------------------------------------

def _upload_eval_zip(
    wandb_run: Any,
    all_prompts: list[EvalPrompt],
    all_rollouts: list[EvalGenerationRollout],
) -> None:
    """Build a zip with all eval parquet tables and upload to wandb."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rel_path = f"evals_after_training/evals_after_training_{ts}.zip"

    dest_path = Path(wandb_run.dir) / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Build parquet tables
    prompts_table = _eval_prompts_to_table(all_prompts)
    generations_table = _eval_generations_to_table(all_rollouts)
    env_responses_table = _eval_env_responses_to_table(all_rollouts)
    tool_calls_table = _eval_tool_calls_to_table(all_rollouts)
    samples_data_table = _samples_data_eval_to_table(all_rollouts)
    metrics_table = _rollouts_metrics_eval_to_table(all_rollouts)
    golden_answers_table = _golden_answers_eval_to_table(all_rollouts)
    info_turns_table = _info_turns_eval_to_table(all_rollouts)
    sample_tags_table = _sample_tags_eval_to_table(all_rollouts)

    steps = [p.step for p in all_prompts]
    metadata: dict[str, Any] = {
        "min_step": min(steps) if steps else 0,
        "max_step": max(steps) if steps else 0,
        "table_schema_versions": dict(table_schema_versions),
    }

    def _write_table(zf: zipfile.ZipFile, name: str, table: pa.Table) -> None:
        buf = io.BytesIO()
        pq.write_table(table, buf)
        zf.writestr(name, buf.getvalue())

    with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_table(zf, "prompts_eval.parquet", prompts_table)
        _write_table(zf, "generations_eval.parquet", generations_table)
        _write_table(zf, "env_responses_eval.parquet", env_responses_table)
        _write_table(zf, "tool_calls_eval.parquet", tool_calls_table)
        _write_table(zf, "samples_data_eval.parquet", samples_data_table)
        _write_table(zf, "rollouts_metrics_eval.parquet", metrics_table)
        _write_table(zf, "golden_answers_eval.parquet", golden_answers_table)
        _write_table(zf, "info_turns_eval.parquet", info_turns_table)
        _write_table(zf, "sample_tags_eval.parquet", sample_tags_table)
        zf.writestr("metadata.json", json.dumps(metadata))

    wandb_run.save(str(dest_path), base_path=wandb_run.dir, policy="now")
    _log.info(f"Uploaded eval results to wandb: {rel_path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

async def _run_eval_async(eval_cfg: EvalStandaloneConfig) -> None:
    """Async core of the standalone eval driver."""
    # Resolve checkpoints first (needed for tokenizer + initial config)
    resolved_checkpoints = _resolve_checkpoints(eval_cfg)
    _log.info(f"Evaluating {len(resolved_checkpoints)} checkpoint(s): "
              f"steps={[c.step for c in resolved_checkpoints]}")

    first_ckpt_path = resolved_checkpoints[0].path

    # Install initial compat config (for Ray init — model path is overwritten per-checkpoint)
    _install_compat_config(eval_cfg, model_path=first_ckpt_path)

    # Load tokenizer — native checkpoints store tokenizer files under hf_meta/
    if eval_cfg.hf_weights:
        tokenizer_path = first_ckpt_path
    else:
        tokenizer_path = os.path.join(first_ckpt_path, "hf_meta")
    _log.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if eval_cfg.chat_template is not None:
        _log.info("Overriding tokenizer chat_template from config")
        tokenizer.chat_template = eval_cfg.chat_template
    elif not getattr(tokenizer, "chat_template", None):
        _log.warning(
            f"Tokenizer has no chat_template. "
            f"Set 'chat_template' in your config to provide one."
        )

    # Init Ray
    _log.info("Initializing Ray cluster")
    init_ray_cluster()

    # Attach to existing wandb run
    parts = eval_cfg.wandb_run_path.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"wandb_run_path must be 'entity/project/run_id', got: {eval_cfg.wandb_run_path!r}"
        )
    wb_entity, wb_project, wb_run_id = parts
    _log.info(f"Attaching to wandb run: {eval_cfg.wandb_run_path}")
    wandb_run = wandb.init(entity=wb_entity, project=wb_project, id=wb_run_id, resume="allow")

    # Parse eval configs and create EvalRunner
    eval_configs = _parse_eval_configs(eval_cfg)
    eval_runner = EvalRunner(eval_configs, tokenizer=tokenizer)
    eval_runner.load_environments()

    # Compute inference topology
    tp_size = max(1, eval_cfg.inference_tensor_parallel_size)
    num_servers = max(1, eval_cfg.inference_num_workers // tp_size)

    # HTTP client for eval requests
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=0)
    http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(1200.0))

    # Accumulate all eval data across checkpoints for a single zip upload
    all_eval_prompts: list[EvalPrompt] = []
    all_eval_rollouts: list[EvalGenerationRollout] = []

    try:
        for ckpt in resolved_checkpoints:
            _log.info(f"Evaluating checkpoint step={ckpt.step} path={ckpt.path}")

            # Convert native checkpoint to HF format if needed
            converted_dir: Path | None = None
            if eval_cfg.hf_weights:
                model_path = ckpt.path
            else:
                converted_dir = Path(ckpt.path + "_hf")
                model_path = _convert_checkpoint(ckpt.path, converted_dir)

            # Update config singleton so vLLM and API requests use this checkpoint
            _install_compat_config(eval_cfg, model_path=model_path)

            # Start fresh vLLM servers with this checkpoint
            inference_group = RayInferenceGroup(
                num_servers=num_servers,
                gpus_per_server=tp_size,
                cpus_per_worker=eval_cfg.ray_inference_cpus_per_worker,
                placement_strategy=eval_cfg.ray_inference_placement_strategy,
                startup_timeout_s=eval_cfg.ray_placement_timeout_s,
                bind_host=eval_cfg.inference_host,
                model=model_path,
            )
            inference_group.launch()
            inference_group.wait_ready()
            server_urls = inference_group.server_urls
            _log.info(f"vLLM servers ready: {server_urls}")

            # Run evals
            eval_max_concurrent = eval_cfg.max_concurrent_samples_per_server
            results = await eval_runner.run_evals(
                eval_configs=eval_configs,
                eval_server_urls=server_urls,
                max_concurrent_per_server=eval_max_concurrent,
                step=ckpt.step,
                model_step=ckpt.step,
                http_client=http_client,
            )

            # Collect eval objects for this checkpoint
            prompts, rollouts = _collect_eval_objects(
                results, step=ckpt.step, model_step=ckpt.step, tokenizer=tokenizer,
            )
            all_eval_prompts.extend(prompts)
            all_eval_rollouts.extend(rollouts)

            # Stop servers before next checkpoint
            _log.info(f"Stopping vLLM servers for step={ckpt.step}")
            inference_group.stop()

            # Clean up converted checkpoint to free disk space
            if converted_dir is not None:
                _cleanup_converted(converted_dir)

        # Upload all results as a single zip
        if all_eval_prompts or all_eval_rollouts:
            _upload_eval_zip(wandb_run, all_eval_prompts, all_eval_rollouts)

    finally:
        await http_client.aclose()
        await eval_runner.close()
        wandb_run.finish()
        if eval_cfg.ray_shutdown_on_exit:
            ray.shutdown()


def run_eval(eval_cfg: EvalStandaloneConfig) -> None:
    """Entry point: run standalone eval (sync wrapper around async core)."""
    setup_logging()
    asyncio.run(_run_eval_async(eval_cfg))

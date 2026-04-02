"""
Eval runner for dedicated eval servers.

Evals use registered evals (from ``telescope/evals/``) or fall back to
registered environments.  They run with their own generation config
(temperature, max_tokens, pass_k) on dedicated eval inference servers.
Results are logged to eval-specific tables, completely separate from training
metrics and rewards.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from telescope.utils import config
from telescope.evals import resolve as resolve_eval
from telescope.environments.base import (
    Environment,
    Sample,
)
from telescope.orchestrator.generate import (
    ContextExhaustedError,
    PromptTooLongError,
    RolloutError,
    SampleLifecycleCallbacks,
    _retry_request,
    get_chat_template_kwargs,
    run_multiturn_rollout,
)
from telescope.utils.tlog import get_logger

_log = get_logger("eval")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PassKEntry:
    """A single pass@k or pass^k specification: which metrics and which k values."""
    metrics: list[str] = field(default_factory=list)
    k: list[int] = field(default_factory=list)


@dataclass
class PassKConfig:
    """Configuration for pass@k (at least 1 correct) and pass^k (all correct) metrics."""
    at_k: PassKEntry = field(default_factory=PassKEntry)
    pow_k: PassKEntry = field(default_factory=PassKEntry)


@dataclass
class EvalConfig:
    """Parsed configuration for a single eval."""
    name: str
    eval_every: int
    pass_k: PassKConfig = field(default_factory=PassKConfig)
    num_samples: int = -1
    separate_eval_samples: bool = False
    kwargs: dict = field(default_factory=dict)
    sampling_params: dict = field(default_factory=dict)


@dataclass
class EvalSampleResult:
    """Result of evaluating a single completion."""
    eval_name: str
    sample_idx: int
    completion_idx: int
    metrics: dict[str, float]
    golden_answers: dict[str, str | None]
    info_turns: list[dict[str, Any]]
    completion_text: str
    prompt_text: str
    sample_tags: dict[str, str] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)
    num_turns: int = 1
    stop_reason: str | None = None
    compute_eval_metrics_time: float = 0.0
    # Inference timing (for logging inference events)
    start_time: float = 0.0
    end_time: float = 0.0
    server_url: str = ""
    prompt_tokens: int = 0
    rollout_tokens: int = 0
    vllm_request_id: str = ""
    max_tokens: int = 0
    system_prompt: str = ""
    env_name: str = ""


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def compute_eval_schedule(
    eval_configs: list[EvalConfig],
    num_steps: int,
) -> dict[int, list[EvalConfig]]:
    """Compute eval schedule: internal 0-indexed step -> list of evals to run.

    Trigger condition: ``(step + 1) % eval_every == 0``.
    So ``eval_every=10`` triggers on internal steps 9, 19, 29, …
    (saved/logged as 1-indexed steps 10, 20, 30, …).
    """
    schedule: dict[int, list[EvalConfig]] = {}
    for ec in eval_configs:
        if ec.eval_every <= 0:
            continue
        for step in range(num_steps):
            if (step + 1) % ec.eval_every == 0:
                schedule.setdefault(step, []).append(ec)
    return schedule


def compute_pass_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    P(at least 1 correct in k draws) = 1 - C(n-c, k) / C(n, k).
    ``n`` = total completions, ``c`` = correct completions, ``k`` = draws.
    """
    if n < k:
        return 0.0 if c == 0 else 1.0
    if c == 0:
        return 0.0
    if c >= n or n - c < k:
        return 1.0
    log_result = 0.0
    for i in range(k):
        log_result += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_result)


def compute_pass_pow_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass^k — P(all k draws correct).

    pass^k = C(c, k) / C(n, k).
    ``n`` = total completions, ``c`` = correct completions, ``k`` = draws.
    """
    if n < k:
        return 1.0 if c >= k else 0.0
    if c < k:
        return 0.0
    if c >= n:
        return 1.0
    log_result = 0.0
    for i in range(k):
        log_result += math.log(c - i) - math.log(n - i)
    return math.exp(log_result)


# ---------------------------------------------------------------------------
# Generation helpers (eval-specific: own temp / max_tokens / n=1 per request)
# ---------------------------------------------------------------------------

async def _generate_eval_completion(
    client: httpx.AsyncClient,
    prompt: str,
    server_url: str,
    sampling_params: dict,
    prompt_token_count: int | None = None,
) -> tuple[dict, float, float, int]:
    """Generate a single eval completion with eval-specific generation params."""
    requested_tokens = sampling_params.get("max_tokens")
    if requested_tokens is None:
        raise ValueError(
            "max_tokens is required in sampling_params but was not set. "
            "Please set max_tokens in your config."
        )
    actual_max_tokens = requested_tokens

    if prompt_token_count is not None:
        available = config.cfg.max_model_len - prompt_token_count
        if available <= 0:
            raise PromptTooLongError(prompt_token_count, config.cfg.max_model_len)
        actual_max_tokens = min(requested_tokens, available)

    data = {
        "model": config.cfg.model,
        "prompt": [prompt],
        "return_token_ids": True,
        "n": 1,
        "logprobs": 1,
        "skip_special_tokens": False,
        "include_stop_str_in_output": False,
        **sampling_params,
        "max_tokens": actual_max_tokens,
    }
    start_time = time.time()

    async def _make():
        resp = await client.post(f"{server_url}/v1/completions", json=data)
        return resp.json()

    result = await _retry_request(_make)
    end_time = time.time()

    if "choices" not in result:
        error_msg = result.get("message", result.get("detail", str(result)))
        raise RolloutError(result.get("type", "unknown"), error_msg, result)

    return result, start_time, end_time, actual_max_tokens


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Loads eval environments, dispatches eval prompts, collects metrics."""

    def __init__(self, eval_configs: list[EvalConfig], tokenizer: Any = None):
        self.configs = eval_configs
        self.tokenizer = tokenizer
        # env name -> (Environment, list[Sample])
        self.envs: dict[str, tuple[Environment, list[Sample]]] = {}
        self._http_client: httpx.AsyncClient | None = None
        # Set during run_evals() for real-time rollout event logging
        self._event_logger: Any | None = None
        self._all_server_urls: list[str] = []
        self._allocate_sample_id: Any | None = None
        self._allocate_group_id: Any | None = None

    # ------------------------------------------------------------------
    # Rollout event helpers
    # ------------------------------------------------------------------

    def _make_eval_lifecycle(
        self, sample_id: int, group_id: int, server_url: str,
    ) -> SampleLifecycleCallbacks | None:
        """Create per-sample lifecycle callbacks for real-time rollout event logging."""
        if self._event_logger is None:
            return None
        server_idx = (
            self._all_server_urls.index(server_url)
            if server_url in self._all_server_urls else -1
        )

        def on_generation_start(generation_idx: int):
            self._event_logger.log_rollout_event(
                event_type="generation", phase="start",
                sample_id=sample_id, server_id=server_idx,
                generation_idx=generation_idx, group_id=group_id,
            )

        def on_generation_end(generation_idx: int):
            self._event_logger.log_rollout_event(
                event_type="generation", phase="end",
                sample_id=sample_id, server_id=server_idx,
                generation_idx=generation_idx, group_id=group_id,
            )

        def on_env_response_start():
            self._event_logger.log_rollout_event(
                event_type="env_response", phase="start",
                sample_id=sample_id, group_id=group_id,
            )

        def on_env_response_end():
            self._event_logger.log_rollout_event(
                event_type="env_response", phase="end",
                sample_id=sample_id, group_id=group_id,
            )

        return SampleLifecycleCallbacks(
            on_generation_start=on_generation_start,
            on_generation_end=on_generation_end,
            on_env_response_start=on_env_response_start,
            on_env_response_end=on_env_response_end,
        )

    def _log_eval_metrics_event(self, phase: str, sample_id: int, group_id: int):
        """Emit an eval_metrics start/end rollout event."""
        if self._event_logger is not None:
            self._event_logger.log_rollout_event(
                event_type="eval_metrics", phase=phase,
                sample_id=sample_id, group_id=group_id,
            )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def load_environments(self):
        """Load and cache eval sources (evals or environments) + their datasets."""
        for ec in self.configs:
            if ec.name in self.envs:
                continue
            _log.info(f"Loading eval source: {ec.name}")
            source = resolve_eval(ec.name, **ec.kwargs)
            source.load_dataset()
            samples = list(source._samples or [])
            for s in samples:
                s.metadata.setdefault("_env_name", getattr(source, "name", ec.name))
            self.envs[ec.name] = (source, samples)
            _log.info(f"  loaded {len(samples)} samples for eval {ec.name}")

    # ------------------------------------------------------------------
    # Prompt prefetch
    # ------------------------------------------------------------------

    def _prefetch_sample_prompt(self, env: Environment, sample: Sample) -> dict:
        """Pre-prepare a prompt for an eval sample to reduce dispatch latency."""
        result: dict[str, Any] = {}

        if getattr(env, "is_multi_turn", False):
            try:
                messages = env.get_initial_prompt(sample, self.tokenizer)
                result["prefetched_messages"] = messages
                if self.tokenizer is not None:
                    chat_kwargs = get_chat_template_kwargs()
                    prompt_str = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_kwargs,
                    )
                    result["prefetched_prompt_str"] = prompt_str
                    result["prefetched_prompt_token_count"] = len(
                        self.tokenizer.encode(prompt_str)
                    )
            except Exception as exc:
                _log.warning(f"Eval multi-turn prefetch failed: {exc}")
        else:
            try:
                chat_kwargs = get_chat_template_kwargs()
                if hasattr(env, "format_prompt") and self.tokenizer is not None:
                    prompt_str = env.format_prompt(
                        sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt),
                        tokenizer=self.tokenizer,
                        chat_template_kwargs=chat_kwargs,
                    )
                else:
                    prompt_str = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
                result["prefetched_prompt_str"] = prompt_str
                if self.tokenizer is not None:
                    result["prefetched_prompt_token_count"] = len(
                        self.tokenizer.encode(prompt_str)
                    )
            except Exception as exc:
                _log.warning(f"Eval single-turn prefetch failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # Run evals for one step
    # ------------------------------------------------------------------

    async def run_evals(
        self,
        eval_configs: list[EvalConfig],
        eval_server_urls: list[str],
        max_concurrent_per_server: int,
        step: int,
        model_step: int,
        http_client: httpx.AsyncClient | None = None,
        event_logger: Any | None = None,
        all_server_urls: list[str] | None = None,
        allocate_sample_id: Any | None = None,
        allocate_group_id: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Run all evals for a given step concurrently on the eval servers.

        Returns a list of per-eval result dicts suitable for logging.

        Args:
            event_logger: EventLogger for real-time rollout event logging.
            all_server_urls: All inference server URLs (for server_id mapping).
            allocate_sample_id: Callable returning a globally unique sample ID.
            allocate_group_id: Callable returning a globally unique group ID.
        """
        # Store event logging state for use by helper methods
        self._event_logger = event_logger
        self._all_server_urls = all_server_urls or []
        self._allocate_sample_id = allocate_sample_id
        self._allocate_group_id = allocate_group_id

        client = http_client or self._http_client
        if client is None:
            limits = httpx.Limits(max_connections=4096, max_keepalive_connections=0)
            self._http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(1200.0))
            client = self._http_client

        total_capacity = max_concurrent_per_server * len(eval_server_urls)
        semaphore = asyncio.Semaphore(total_capacity)
        server_cycle = _cycle(eval_server_urls)

        all_tasks: list[asyncio.Task] = []
        eval_results: list[dict[str, Any]] = []

        prefetch_enabled = config.cfg.enable_prompt_prefetch

        for ec in eval_configs:
            env, samples = self.envs[ec.name]
            num = min(ec.num_samples, len(samples)) if ec.num_samples > 0 else len(samples)
            selected = samples[:num]

            # Prefetch prompts for all selected samples (once per sample)
            prefetched: dict[int, dict] = {}
            if prefetch_enabled:
                for idx, sample in enumerate(selected):
                    prefetched[idx] = self._prefetch_sample_prompt(env, sample)
                _log.info(f"Prefetched {len(prefetched)} eval prompts for {ec.name}")

            per_eval_futures: list[asyncio.Future] = []

            all_k = (ec.pass_k.at_k.k or []) + (ec.pass_k.pow_k.k or [])
            max_completions = max(all_k) if all_k else 1
            for sample_idx, sample in enumerate(selected):
                # One group_id per unique prompt (shared across pass@k completions)
                group_id = self._allocate_group_id() if self._allocate_group_id else -1
                for comp_idx in range(max_completions):
                    sample_id = self._allocate_sample_id() if self._allocate_sample_id else -1
                    fut = asyncio.get_event_loop().create_future()
                    per_eval_futures.append(fut)

                    task = asyncio.create_task(
                        self._run_one_eval_completion(
                            client=client,
                            env=env,
                            sample=sample,
                            sample_idx=sample_idx,
                            completion_idx=comp_idx,
                            eval_config=ec,
                            semaphore=semaphore,
                            server_cycle=server_cycle,
                            result_future=fut,
                            prefetched_data=prefetched.get(sample_idx, {}),
                            sample_id=sample_id,
                            group_id=group_id,
                        )
                    )
                    all_tasks.append(task)

            # Gather all completion results for this eval
            results_for_eval = asyncio.ensure_future(
                self._collect_eval_results(ec, per_eval_futures, step, model_step)
            )
            all_tasks.append(results_for_eval)

        # Wait for everything
        done = await asyncio.gather(*all_tasks, return_exceptions=True)
        # The _collect_eval_results tasks are the last len(eval_configs) entries
        for item in done:
            if isinstance(item, dict):
                eval_results.append(item)
            elif isinstance(item, Exception):
                _log.warning(f"Eval task failed: {item}")

        # Clear event logging state
        self._event_logger = None
        self._all_server_urls = []
        self._allocate_sample_id = None
        self._allocate_group_id = None

        return eval_results

    async def _run_one_eval_completion(
        self,
        client: httpx.AsyncClient,
        env: Environment,
        sample: Sample,
        sample_idx: int,
        completion_idx: int,
        eval_config: EvalConfig,
        semaphore: asyncio.Semaphore,
        server_cycle,
        result_future: asyncio.Future,
        prefetched_data: dict | None = None,
        sample_id: int = -1,
        group_id: int = -1,
    ):
        """Run a single eval completion, respecting concurrency limits."""
        async with semaphore:
            server_url = next(server_cycle)
            try:
                result = await self._eval_single_sample(
                    client=client,
                    env=env,
                    sample=sample,
                    sample_idx=sample_idx,
                    completion_idx=completion_idx,
                    eval_config=eval_config,
                    server_url=server_url,
                    prefetched_data=prefetched_data,
                    sample_id=sample_id,
                    group_id=group_id,
                )
                result_future.set_result(result)
            except Exception as exc:
                _log.warning(f"Eval completion failed ({eval_config.name} sample={sample_idx} comp={completion_idx}): {exc}")
                result_future.set_result(None)

    async def _eval_single_sample(
        self,
        client: httpx.AsyncClient,
        env: Environment,
        sample: Sample,
        sample_idx: int,
        completion_idx: int,
        eval_config: EvalConfig,
        server_url: str,
        prefetched_data: dict | None = None,
        sample_id: int = -1,
        group_id: int = -1,
    ) -> EvalSampleResult | None:
        """Evaluate one completion for one sample."""

        if getattr(env, "is_multi_turn", False):
            return await self._eval_multiturn(
                client, env, sample, sample_idx, completion_idx, eval_config, server_url,
                prefetched_data=prefetched_data,
                sample_id=sample_id,
                group_id=group_id,
            )

        # Single-turn eval
        lifecycle = self._make_eval_lifecycle(sample_id, group_id, server_url)
        system_prompt = env.system_prompt or ""
        actual_env_name = getattr(env, "name", eval_config.name) or eval_config.name

        if prefetched_data and "prefetched_prompt_str" in prefetched_data:
            prompt_str = prefetched_data["prefetched_prompt_str"]
            prompt_token_count = prefetched_data.get("prefetched_prompt_token_count")
        else:
            chat_kwargs = get_chat_template_kwargs()
            if hasattr(env, "format_prompt") and self.tokenizer is not None:
                prompt_str = env.format_prompt(
                    sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt),
                    tokenizer=self.tokenizer,
                    chat_template_kwargs=chat_kwargs,
                )
            else:
                prompt_str = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
            prompt_token_count = len(self.tokenizer.encode(prompt_str)) if self.tokenizer else None

        if lifecycle and lifecycle.on_generation_start:
            lifecycle.on_generation_start(0)
        try:
            result, req_start_time, req_end_time, actual_max_tokens = await _generate_eval_completion(
                client, prompt_str, server_url,
                sampling_params=eval_config.sampling_params,
                prompt_token_count=prompt_token_count,
            )
        except (PromptTooLongError, ContextExhaustedError, RolloutError) as exc:
            if lifecycle and lifecycle.on_generation_end:
                lifecycle.on_generation_end(0)
            _log.warning(f"Eval generation error: {exc}")
            return None
        if lifecycle and lifecycle.on_generation_end:
            lifecycle.on_generation_end(0)

        choice = result["choices"][0]
        completion_text = choice.get("text", "")
        finish_reason = choice.get("finish_reason", "")
        rollout_tokens = len(choice.get("token_ids", []))
        usage = result.get("usage", {})
        actual_prompt_tokens = usage.get("prompt_tokens", prompt_token_count or 0)
        vllm_request_id = result.get("id", "")

        eos_token = getattr(env, "eos_token", "")
        self._log_eval_metrics_event("start", sample_id, group_id)
        metrics_start = time.time()
        eval_result = await env.compute_eval_metrics(completion_text, sample, eos_token)
        compute_eval_metrics_time = time.time() - metrics_start
        self._log_eval_metrics_event("end", sample_id, group_id)

        prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
        turns = [{
            "turn_order": 0,
            "turn_type": "model",
            "content": completion_text,
            "tokens": rollout_tokens,
            "stop_reason": finish_reason,
            "environment_response_time": 0.0,
        }]
        return EvalSampleResult(
            eval_name=eval_config.name,
            sample_idx=sample_idx,
            completion_idx=completion_idx,
            metrics=eval_result.metrics,
            golden_answers=eval_result.golden_answers,
            info_turns=eval_result.info_turns,
            sample_tags=eval_result.sample_tags,
            completion_text=completion_text,
            prompt_text=prompt_text,
            turns=turns,
            num_turns=1,
            stop_reason=finish_reason,
            compute_eval_metrics_time=compute_eval_metrics_time,
            start_time=req_start_time,
            end_time=req_end_time,
            server_url=server_url,
            prompt_tokens=actual_prompt_tokens,
            rollout_tokens=rollout_tokens,
            vllm_request_id=vllm_request_id,
            max_tokens=actual_max_tokens,
            system_prompt=system_prompt,
            env_name=actual_env_name,
        )

    async def _eval_multiturn(
        self,
        client: httpx.AsyncClient,
        env: Environment,
        sample: Sample,
        sample_idx: int,
        completion_idx: int,
        eval_config: EvalConfig,
        server_url: str,
        prefetched_data: dict | None = None,
        sample_id: int = -1,
        group_id: int = -1,
    ) -> EvalSampleResult | None:
        """Evaluate one multi-turn completion for one sample.

        Uses ``run_multiturn_rollout`` in non-interleaved mode (no token
        tracking since we're not training on this data).
        """
        assert getattr(env, "is_multi_turn", False), f"{env} is not multi-turn"

        lifecycle = self._make_eval_lifecycle(sample_id, group_id, server_url)
        system_prompt = env.system_prompt or ""
        actual_env_name = getattr(env, "name", eval_config.name) or eval_config.name

        req_start_time = time.time()
        state, request_timings = await run_multiturn_rollout(
            client, env, sample, server_url, self.tokenizer,
            prefetched_messages=prefetched_data.get("prefetched_messages") if prefetched_data else None,
            prefetched_prompt_str=prefetched_data.get("prefetched_prompt_str") if prefetched_data else None,
            prefetched_prompt_token_count=prefetched_data.get("prefetched_prompt_token_count") if prefetched_data else None,
            sampling_params=eval_config.sampling_params,
            interleaved=False,
            lifecycle=lifecycle,
        )
        req_end_time = time.time()

        if state.error or state.num_turns == 0:
            return None

        eos_token = getattr(env, "eos_token", "")
        self._log_eval_metrics_event("start", sample_id, group_id)
        metrics_start = time.time()
        eval_result = await env.compute_eval_metrics(state, eos_token)
        compute_eval_metrics_time = time.time() - metrics_start
        self._log_eval_metrics_event("end", sample_id, group_id)

        total_prompt_tokens = 0
        total_rollout_tokens = 0
        timing_by_turn: dict[int, dict] = {}
        for rt in request_timings:
            turn_num = rt.get("turn", 0)
            timing_by_turn[turn_num] = rt
            total_prompt_tokens += rt.get("prompt_tokens", 0)
            total_rollout_tokens += rt.get("rollout_tokens", 0)

        vllm_request_id = ""
        eval_max_tokens = eval_config.sampling_params.get("max_tokens")
        if eval_max_tokens is None:
            raise ValueError(
                f"max_tokens is required in sampling_params for eval '{eval_config.name}' "
                "but was not set. Please set max_tokens in your config."
            )
        last_max_tokens = eval_max_tokens
        if request_timings:
            vllm_request_id = request_timings[-1].get("vllm_request_id", "")
            last_max_tokens = request_timings[-1].get("max_tokens", eval_max_tokens)

        completion_text = ""
        turns: list[dict[str, Any]] = state.custom.get("_logged_turns", [])
        for turn in turns:
            if turn.get("turn_type") == "model":
                completion_text += turn.get("content", "") + "\n"
        completion_text = completion_text.strip()

        prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)

        return EvalSampleResult(
            eval_name=eval_config.name,
            sample_idx=sample_idx,
            completion_idx=completion_idx,
            metrics=eval_result.metrics,
            golden_answers=eval_result.golden_answers,
            info_turns=eval_result.info_turns,
            sample_tags=eval_result.sample_tags,
            completion_text=completion_text,
            prompt_text=prompt_text,
            turns=turns,
            num_turns=state.num_turns,
            stop_reason=state.stop_reason,
            compute_eval_metrics_time=compute_eval_metrics_time,
            start_time=req_start_time,
            end_time=req_end_time,
            server_url=server_url,
            prompt_tokens=total_prompt_tokens,
            rollout_tokens=total_rollout_tokens,
            vllm_request_id=vllm_request_id,
            max_tokens=last_max_tokens,
            system_prompt=system_prompt,
            env_name=actual_env_name,
        )

    async def _collect_eval_results(
        self,
        eval_config: EvalConfig,
        futures: list[asyncio.Future],
        step: int,
        model_step: int,
    ) -> dict[str, Any]:
        """Wait for all completions of one eval and aggregate metrics."""
        results: list[EvalSampleResult | None] = []
        for fut in futures:
            r = await fut
            results.append(r)

        valid = [r for r in results if r is not None]
        pk = eval_config.pass_k
        at_k_entry = pk.at_k
        pow_k_entry = pk.pow_k
        all_k = (at_k_entry.k or []) + (pow_k_entry.k or [])
        max_completions = max(all_k) if all_k else 1

        # Per-sample aggregation
        by_sample: dict[int, list[EvalSampleResult]] = {}
        for r in valid:
            by_sample.setdefault(r.sample_idx, []).append(r)

        # Collect all metrics keys
        all_metric_keys: set[str] = set()
        for r in valid:
            all_metric_keys.update(r.metrics.keys())

        # Aggregate scalar averages across all completions
        avg_metrics: dict[str, float] = {}
        for key in all_metric_keys:
            values = [r.metrics[key] for r in valid if key in r.metrics]
            if values:
                avg_metrics[key] = sum(values) / len(values)

        # pass@k and pass^k: compute per-sample, inject into every completion
        at_k_metrics = set(at_k_entry.metrics) & all_metric_keys if at_k_entry.metrics else set()
        pow_k_metrics = set(pow_k_entry.metrics) & all_metric_keys if pow_k_entry.metrics else set()
        if at_k_metrics or pow_k_metrics:
            for sample_idx, sample_results in by_sample.items():
                sample_pk: dict[str, float] = {}
                if at_k_metrics:
                    for k in sorted(at_k_entry.k):
                        first_k = sample_results[:k]
                        for key in at_k_metrics:
                            values = [r.metrics.get(key, 0.0) for r in first_k]
                            n = len(values)
                            c = sum(1 for v in values if v > 0)
                            sample_pk[f"pass@{k}/{key}"] = compute_pass_k(n, c, k)
                if pow_k_metrics:
                    for k in sorted(pow_k_entry.k):
                        first_k = sample_results[:k]
                        for key in pow_k_metrics:
                            values = [r.metrics.get(key, 0.0) for r in first_k]
                            n = len(values)
                            c = sum(1 for v in values if v > 0)
                            sample_pk[f"pass^{k}/{key}"] = compute_pass_pow_k(n, c, k)
                for r in sample_results:
                    r.metrics.update(sample_pk)

        return {
            "eval_name": eval_config.name,
            "step": step,
            "model_step": model_step,
            "num_samples": len(by_sample),
            "num_completions": len(valid),
            "max_completions": max_completions,
            "avg_metrics": avg_metrics,
            "sample_results": valid,
        }

    async def close(self):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


def _cycle(items):
    """Infinite cycle iterator over a list."""
    import itertools
    return itertools.cycle(items)

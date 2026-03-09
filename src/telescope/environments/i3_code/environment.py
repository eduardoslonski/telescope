"""
i3-code environment implementation.

Adapted from https://app.primeintellect.ai/dashboard/environments/primeintellect/code-env

Coding environment using the INTELLECT-3-RL code dataset with sandboxed
test execution.  Supports multiple sandbox providers via the ``provider``
argument (e.g. ``"daytona"``, ``"prime"``, ``"modal"``, ``"e2b"``).

A provider must be specified — install the corresponding package:
    daytona: uv add daytona-sdk
    prime:   uv add prime-sandboxes
    modal:   uv add modal
    e2b:     uv add e2b
"""
REQUIRED_PACKAGES = []
OPTIONAL_PACKAGES = ["daytona-sdk", "prime-sandboxes", "modal", "e2b"]

import asyncio
import atexit
import json
import logging
import os
import random
import signal
import time
from typing import cast

from datasets import Dataset, load_dataset

from telescope.environments.base import EvalMetricsResult, Sample, RewardResult, SingleTurnEnvironment
from telescope.environments.parsers import strip_think_tags
from telescope.environments._sandbox import (
    GenericSandboxPool,
    SandboxConfig,
    SandboxError,
    SandboxNotRunningError,
    get_provider,
)

from .deepcoder_utils import extract_code_from_model
from .verification_utils import TestRunResult, run_test_cases


logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_PROMPT = "Solve the programming task below in a Python markdown code block."

DEFAULT_DOCKER_IMAGE = "amancevice/pandas:slim"

# Provider-specific defaults
_PROVIDER_DEFAULTS = {
    "prime": {
        "workspace_path": "/sandbox-workspace",
        "num_workers": 32,
        "check_fd_limit": True,
    },
    "modal": {
        "workspace_path": "/tmp",
        "num_workers": 1,
        "check_fd_limit": False,
    },
    "daytona": {
        "workspace_path": "/tmp",
        "num_workers": 32,
        "check_fd_limit": False,
    },
    "e2b": {
        "workspace_path": "/tmp",
        "num_workers": 32,
        "check_fd_limit": False,
    },
}


def _check_file_descriptor_limit(min_limit: int = 65536) -> None:
    """Early check for available file descriptors (sandbox usage needs many)."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception as e:
        raise RuntimeError(f"Could not check file descriptor limit (RLIMIT_NOFILE): {e}")
    if soft < min_limit:
        raise RuntimeError(
            f"File descriptor limit (RLIMIT_NOFILE) is set to {soft}. "
            f"This is likely not high enough for high-concurrency sandbox usage! "
            f"Consider increasing it to at least {min_limit} via "
            f"`ulimit -n {min_limit}` or your system configuration."
        )


def _process_test_cases(tests: dict, max_num_tests: int = 15) -> dict:
    """Select and serialize test cases."""
    total_tests = len(tests["inputs"])
    if total_tests > max_num_tests:
        selected_indices = random.sample(range(total_tests), max_num_tests)
    else:
        selected_indices = range(total_tests)
    inputs = [json.dumps(tests["inputs"][i]) for i in selected_indices]
    outputs = [json.dumps(tests["outputs"][i]) for i in selected_indices]
    return {**tests, "inputs": inputs, "outputs": outputs}


class I3CodeEnvironment(SingleTurnEnvironment):
    """
    Environment for INTELLECT-3 coding tasks with sandboxed execution.

    Uses the PrimeIntellect/INTELLECT-3-RL dataset (code subset).
    Reward is based on passing all test cases in a sandboxed environment.

    Test execution flow:
    1. Extract code from model completion (markdown code block)
    2. Acquire a sandbox from the pool
    3. Upload and run test cases in the sandbox
    4. Return reward based on pass rate (1.0 if all pass, 0.0 otherwise)

    The sandbox pool is started lazily on first compute_reward call and
    cleaned up via atexit/signal handlers.

    Supports providers: ``"prime"`` (default), ``"modal"``, ``"daytona"``, ``"e2b"``.
    """

    def __init__(
        self,
        # Dataset options
        dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
        dataset_subset: str = "code",
        dataset_split: str = "train",
        difficulty_key: str | None = "avg@8_qwen3_4b_instruct_2507",
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        max_num_tests: int = 15,
        timeout_per_test: int = 10,
        random_seed: int | None = 42,
        # Provider selection
        provider: str | None = None,
        provider_kwargs: dict | None = None,
        # Sandbox options (provider-agnostic)
        pool_size: int = 10,
        max_concurrent_creates: int = 100,
        timeout_minutes: int = 60,
        environment_vars: dict[str, str] | None = None,
        # Sandbox resource options
        docker_image: str | None = None,
        cpu: float = 1.0,
        memory_mb: int = 2048,
        disk_size_gb: int = 3,
        gpu_count: int = 0,
        team_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            system_prompt=None,
            instruction_prompt=DEFAULT_INSTRUCTION_PROMPT,
            **kwargs,
        )
        # Dataset config
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split
        self.difficulty_key = difficulty_key
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.max_num_tests = max_num_tests
        self.timeout_per_test = timeout_per_test
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)

        # ── Provider setup ────────────────────────────────────────────
        if provider is None:
            raise ValueError(
                "A sandbox provider is required for i3_code.\n"
                "Set it in your config YAML:\n"
                "  environments:\n"
                "    - name: \"i3_code\"\n"
                "      kwargs:\n"
                "        provider: \"daytona\"  # or prime, modal, e2b, ...\n\n"
                "Then install the corresponding package:\n"
                "  daytona  ->  uv add daytona-sdk\n"
                "  prime    ->  uv add prime-sandboxes\n"
                "  modal    ->  uv add modal\n"
                "  e2b      ->  uv add e2b"
            )
        self._provider_name = provider
        provider_defaults = _PROVIDER_DEFAULTS.get(provider, {})
        self._workspace_path = provider_defaults.get("workspace_path", "/tmp")
        self._num_workers = provider_defaults.get("num_workers", 32)

        if provider_defaults.get("check_fd_limit", False):
            _check_file_descriptor_limit()

        # Resolve image
        if docker_image is None:
            docker_image = os.getenv("DEFAULT_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
        resolved_image = docker_image or ""

        self._provider = get_provider(provider, **(provider_kwargs or {}))

        # Build sandbox config
        sandbox_config = SandboxConfig(
            image=resolved_image,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            timeout_seconds=timeout_minutes * 60,
            environment_vars=environment_vars or {},
            name="i3-code-env",
            extra={
                "team_id": team_id,
                "start_command": "tail -f /dev/null",
            },
        )

        self.sandbox_pool = GenericSandboxPool(
            provider=self._provider,
            config=sandbox_config,
            pool_size=pool_size,
            max_concurrent_creates=max_concurrent_creates,
        )

        # Limit concurrent sandbox test executions to prevent overwhelming the
        # sandbox API.
        self._max_concurrent_sandbox_ops = min(pool_size, 400)
        self._sandbox_ops_semaphore: asyncio.Semaphore | None = None

        # Register cleanup handlers
        atexit.register(self._cleanup_sandboxes)
        # signal.signal() only works from the main thread; the orchestrator
        # may construct environments in a worker thread, so guard the call.
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(
                signal.SIGINT,
                lambda sig, frame: (
                    self._cleanup_sandboxes(),
                    signal.default_int_handler(sig, frame),
                ),
            )
            signal.signal(signal.SIGTERM, lambda _, __: (self._cleanup_sandboxes(), exit(143)))

    metrics_ranges = {
        "passed": {"min": 0, "max": 1},
        "pass_rate": {"min": 0, "max": 1},
        "sandbox_error": {"min": 0, "max": 1},
        "num_test_cases": {"min": 0, "max": 15},
        "num_timeouts": {"min": 0, "max": 15},
        "num_errors": {"min": 0, "max": 15},
        "retry_count": {"min": 0, "max": 3},
        "failed_attempts_time": {"min": 0, "max": 600},
        # Timing metrics (seconds)
        "ops_semaphore_wait_time": {"min": 0, "max": 60},
        "sandbox_acquire_time": {"min": 0, "max": 600},
        "test_time": {"min": 0, "max": 300},
        "bundle_build_time": {"min": 0, "max": 30},
        "upload_semaphore_wait_time": {"min": 0, "max": 60},
        "bundle_upload_time": {"min": 0, "max": 60},
        "bundle_extract_time": {"min": 0, "max": 60},
        "bundle_total_time": {"min": 0, "max": 120},
        "tests_execution_time": {"min": 0, "max": 300},
        "slowest_test_time": {"min": 0, "max": 30},
        "fastest_test_time": {"min": 0, "max": 30},
        "median_test_time": {"min": 0, "max": 30},
        "total_reward_time": {"min": 0, "max": 600},
        # Code metrics
        "generated_code_chars": {"min": 0, "max": 50000},
    }

    # Max chars for info_turns text values (keep parquet files manageable)
    _MAX_INFO_VALUE_CHARS: int = 4000

    # Progress tracking (shared across all compute_reward calls)
    _reward_count: int = 0
    _reward_passed: int = 0
    _reward_no_code: int = 0
    _reward_errors: int = 0
    _progress_log_interval: int = 10  # log every N completions

    @property
    def name(self) -> str:
        return f"i3-code-{self._provider_name}"

    # ── Dataset loading ────────────────────────────────────────────────

    def load_dataset(
        self,
        num_samples: int = -1,
        shuffle: bool = False,
        seed: int = 42,
        **kwargs,
    ) -> list[Sample]:
        """
        Load the INTELLECT-3-RL code dataset.

        Each sample contains the coding problem and verification info
        (test case inputs/outputs) stored in metadata.

        Args:
            num_samples: Number of samples to load. -1 for all.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.

        Returns:
            List of Sample objects.
        """
        logger.info(f"Loading {self.dataset_name}/{self.dataset_subset}...")

        dataset = cast(
            Dataset,
            load_dataset(self.dataset_name, self.dataset_subset, split=self.dataset_split),
        )

        # Filter by difficulty if key specified
        if self.difficulty_key is not None:
            if self.difficulty_key in dataset.column_names:
                difficulty_key = self.difficulty_key
                min_diff = self.min_difficulty
                max_diff = self.max_difficulty
                dataset = dataset.filter(
                    lambda x, _k=difficulty_key, _lo=min_diff, _hi=max_diff: (
                        _lo <= x.get(_k, 0) <= _hi
                    )
                )
            else:
                logger.warning(
                    f"Difficulty key '{self.difficulty_key}' not found in dataset columns "
                    f"({dataset.column_names}). Skipping difficulty filter."
                )

        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Convert to samples
        samples = []
        for idx, example in enumerate(dataset):
            sample = self._process_example(example, idx)
            if sample is not None:
                samples.append(sample)

        self._samples = samples
        self._dataset = dataset

        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples

    def _process_example(self, example: dict, idx: int) -> Sample | None:
        """Process a single dataset example into a Sample."""
        try:
            info = json.loads(example["info"])
            tests = json.loads(info["tests"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Skipping example {idx}: failed to parse info/tests: {e}")
            return None

        processed_tests = _process_test_cases(tests, max_num_tests=self.max_num_tests)

        verification_info = {
            "fn_name": processed_tests.get("fn_name"),
            "test_case_inputs": processed_tests["inputs"],
            "test_case_outputs": processed_tests["outputs"],
            "timeout": self.timeout_per_test,
        }

        return Sample(
            prompt=example["question"],  # Raw question, formatted at rollout time
            answer="",  # Code tasks use test execution, not answer matching
            metadata={
                "question": example["question"],
                "subset": self.name.replace("-", "_").replace(" ", "_"),
                "subset_idx": idx,
                "source": info.get("source", ""),
                "verification_info": verification_info,
                "num_test_cases": len(processed_tests["inputs"]),
            },
        )

    # ── Info turns builder ──────────────────────────────────────────────

    def _build_info_turns(
        self,
        generated_code: str,
        test_run: TestRunResult | None = None,
        error_msg: str | None = None,
    ) -> list[dict]:
        """Build info_turns list from test execution results.

        Produces structured per-turn info dicts for the logging pipeline:
          - generated_code: the extracted code that was executed
          - test_summary: pass/fail summary with counts
          - passed_tests: details of passed tests (input, output, expected)
          - failed_tests: details of failed tests (input, stdout vs expected, stderr)
          - runner_stderr: runner-level stderr (import errors, crashes)
          - error: sandbox/infrastructure error message

        All values are truncated to _MAX_INFO_VALUE_CHARS.
        """
        limit = self._MAX_INFO_VALUE_CHARS
        info: list[dict] = []

        def _add(key: str, value: str, info_type: str = "text") -> None:
            if value:
                info.append({
                    "turn_order": 0,
                    "info_key": key,
                    "info_value": value[:limit],
                    "info_type": info_type,
                })

        # 1. Generated code
        _add("generated_code", generated_code, info_type="code")

        # 2. Infrastructure error (if any)
        if error_msg:
            _add("error", error_msg, info_type="stderr")

        if test_run is None:
            return info

        # 3. Test summary
        total = len(test_run.raw_results)
        passed_count = sum(1 for r in test_run.raw_results if r is True)
        failed_count = sum(1 for r in test_run.raw_results if r is False)
        error_count = test_run.num_errors
        timeout_count = test_run.num_timeouts
        summary_parts = [f"{passed_count}/{total} passed"]
        if failed_count:
            summary_parts.append(f"{failed_count} failed")
        if error_count:
            summary_parts.append(f"{error_count} errors")
        if timeout_count:
            summary_parts.append(f"{timeout_count} timeouts")
        _add("test_summary", " | ".join(summary_parts))

        # 4. Split test details into passed and failed
        passed_details: list[dict] = []
        failed_details: list[dict] = []
        for detail in test_run.test_details:
            if detail.get("passed"):
                passed_details.append(detail)
            else:
                failed_details.append(detail)

        # Helper to format the header line with test index, status, and time
        def _header(idx, status: str, detail: dict) -> str:
            t = detail.get("time")
            time_str = f" ({t:.3f}s)" if t is not None else ""
            return f"--- test {idx} [{status}]{time_str} ---"

        # 5. Passed test details (show input, actual output, expected)
        if passed_details:
            lines = []
            for detail in passed_details:
                idx = detail.get("i", "?")
                parts = [_header(idx, "PASSED", detail)]
                if "input" in detail and detail["input"]:
                    parts.append(f"input: {detail['input']}")
                if "stdout" in detail and detail["stdout"]:
                    parts.append(f"stdout: {detail['stdout']}")
                if "actual_result" in detail and detail["actual_result"]:
                    parts.append(f"result: {detail['actual_result']}")
                if "expected" in detail and detail["expected"]:
                    parts.append(f"expected: {detail['expected']}")
                lines.append("\n".join(parts))
            _add("passed_tests", "\n\n".join(lines))

        # 6. Failed test details (show input, actual vs expected, stderr)
        if failed_details:
            lines = []
            for detail in failed_details:
                idx = detail.get("i", "?")
                if detail.get("timeout"):
                    parts = [_header(idx, "TIMEOUT", detail)]
                    if "input" in detail and detail["input"]:
                        parts.append(f"input: {detail['input']}")
                    lines.append("\n".join(parts))
                    continue
                if detail.get("error"):
                    parts = [_header(idx, "ERROR", detail)]
                    if "input" in detail and detail["input"]:
                        parts.append(f"input: {detail['input']}")
                    parts.append(f"error: {detail['error']}")
                    lines.append("\n".join(parts))
                    continue
                parts = [_header(idx, "FAILED", detail)]
                if "input" in detail and detail["input"]:
                    parts.append(f"input: {detail['input']}")
                if "stderr" in detail and detail["stderr"]:
                    parts.append(f"stderr: {detail['stderr']}")
                if "stdout" in detail and detail["stdout"]:
                    parts.append(f"actual stdout: {detail['stdout']}")
                if "actual_result" in detail and detail["actual_result"]:
                    parts.append(f"actual result: {detail['actual_result']}")
                if "expected" in detail and detail["expected"]:
                    parts.append(f"expected: {detail['expected']}")
                lines.append("\n".join(parts))
            _add("failed_tests", "\n\n".join(lines))

        # 7. Runner stderr (import errors, syntax errors, runner crashes)
        if test_run.runner_stderr:
            _add("runner_stderr", test_run.runner_stderr, info_type="stderr")

        return info

    # ── Eval metrics (async — delegates to async compute_reward) ─────

    async def compute_eval_metrics(
        self,
        completion: str,
        sample: Sample,
        eos_token: str = "",
    ) -> EvalMetricsResult:
        reward_result = await self.compute_reward(completion, sample, eos_token)
        return EvalMetricsResult(
            metrics=reward_result.sample_metrics,
            golden_answers=reward_result.golden_answers,
            info_turns=reward_result.info_turns,
        )

    # ── Reward computation (async — sandbox test execution) ───────────

    async def compute_reward(
        self,
        completion: str,
        sample: Sample,
        eos_token: str = "",
    ) -> RewardResult:
        """
        Compute reward by executing generated code against test cases in a sandbox.

        This is an async method (the framework supports both sync and async
        compute_reward via inspect.isawaitable).

        Flow:
        1. Extract code from completion (markdown code block)
        2. Acquire sandbox from pool
        3. Run test cases
        4. Return reward (1.0 if all tests pass, 0.0 otherwise)

        Emits detailed timing metrics in sample_metrics for debugging slow samples.
        """
        reward_start = time.perf_counter()

        # Strip EOS token if present
        if eos_token and completion.endswith(eos_token):
            completion = completion[: -len(eos_token)]

        # Extract code — strict think parser logic:
        # If <think> is present but </think> is missing, treat as incomplete
        if "<think>" in completion and "</think>" not in completion:
            generated_code = ""
        else:
            text = strip_think_tags(completion)
            generated_code = extract_code_from_model(text)

        if not generated_code:
            I3CodeEnvironment._reward_count += 1
            I3CodeEnvironment._reward_no_code += 1
            self._log_progress()
            return RewardResult(
                total_reward=0.0,
                sample_metrics={
                    "passed": 0.0,
                    "pass_rate": 0.0,
                    "num_test_cases": float(sample.metadata.get("num_test_cases", 0)),
                    "sandbox_error": 0.0,
                    "total_reward_time": time.perf_counter() - reward_start,
                    "generated_code_chars": 0.0,
                },
                info_turns=self._build_info_turns(
                    "", error_msg="No code extracted from completion",
                ),
            )

        # Ensure sandbox pool is started (idempotent)
        await self.sandbox_pool.start()

        verification_info = sample.metadata["verification_info"]
        num_tests = len(verification_info.get("test_case_inputs", []))
        code_chars = float(len(generated_code))

        def _error_result(
            *,
            acquire_time: float = 0.0,
            retry_count: int = 0,
            test_run: TestRunResult | None = None,
            failed_time: float = 0.0,
            error_msg: str = "",
        ) -> RewardResult:
            """Build full RewardResult for error/failure cases (includes info_turns)."""
            m: dict[str, float] = {
                "passed": 0.0,
                "pass_rate": 0.0,
                "num_test_cases": float(num_tests),
                "sandbox_error": 1.0,
                "ops_semaphore_wait_time": cumulative_semaphore_wait_time,
                "sandbox_acquire_time": acquire_time,
                "retry_count": float(retry_count),
                "failed_attempts_time": failed_time,
                "total_reward_time": time.perf_counter() - reward_start,
                "generated_code_chars": code_chars,
            }
            if test_run is not None:
                m.update(test_run.timings)
                m["num_timeouts"] = float(test_run.num_timeouts)
                m["num_errors"] = float(test_run.num_errors)
            return RewardResult(
                total_reward=0.0,
                sample_metrics=m,
                info_turns=self._build_info_turns(
                    generated_code,
                    test_run=test_run,
                    error_msg=error_msg or None,
                ),
            )

        # Lazily create the concurrency semaphore in the event loop context
        if self._sandbox_ops_semaphore is None:
            self._sandbox_ops_semaphore = asyncio.Semaphore(self._max_concurrent_sandbox_ops)
            logger.info(
                f"Created sandbox ops semaphore (max_concurrent={self._max_concurrent_sandbox_ops})"
            )

        # Retry logic: if a sandbox fails, remove it and retry with a new one
        max_retries = 3
        cumulative_acquire_time = 0.0
        cumulative_semaphore_wait_time = 0.0
        failed_attempts_time = 0.0  # Total wall time wasted on failed attempts
        for attempt in range(max_retries):
          semaphore_wait_start = time.perf_counter()
          async with self._sandbox_ops_semaphore:
            cumulative_semaphore_wait_time += time.perf_counter() - semaphore_wait_start
            attempt_start = time.perf_counter()
            try:
                logger.debug(
                    f"[{sample.metadata.get('subset_idx')}] Acquiring sandbox "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                acquire_start = time.perf_counter()
                handle = await self.sandbox_pool.acquire(timeout=600.0)
                acquire_time = time.perf_counter() - acquire_start
                cumulative_acquire_time += acquire_time
                logger.debug(
                    f"[{sample.metadata.get('subset_idx')}] Acquired sandbox {handle.id} "
                    f"in {acquire_time:.2f}s"
                )

                try:
                    test_start = time.perf_counter()

                    # Overall timeout for test execution (upload + all tests).
                    _TEST_TIMEOUT = 180.0  # 3 minutes
                    try:
                        test_run = await asyncio.wait_for(
                            run_test_cases(
                                generated_code,
                                verification_info,
                                self._provider,
                                handle,
                                num_workers=self._num_workers,
                                workspace_path=self._workspace_path,
                            ),
                            timeout=_TEST_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[{sample.metadata.get('subset_idx')}] Test execution timed out "
                            f"after {_TEST_TIMEOUT:.0f}s in sandbox {handle.id} "
                            f"(attempt {attempt + 1}/{max_retries}) — removing sandbox"
                        )
                        try:
                            await self.sandbox_pool.remove(handle)
                        except Exception:
                            pass
                        failed_attempts_time += time.perf_counter() - attempt_start

                        if attempt == max_retries - 1:
                            logger.error(
                                f"[{sample.metadata.get('subset_idx')}] All {max_retries} "
                                f"sandbox attempts timed out — giving up"
                            )
                            I3CodeEnvironment._reward_count += 1
                            I3CodeEnvironment._reward_errors += 1
                            self._log_progress()
                            return _error_result(
                                acquire_time=cumulative_acquire_time,
                                retry_count=attempt + 1,
                                failed_time=failed_attempts_time,
                                error_msg="All sandbox attempts timed out",
                            )
                        continue

                    test_time = time.perf_counter() - test_start

                    if not test_run.results:
                        # All tests failed due to sandbox infrastructure errors
                        logger.warning(
                            f"All test cases failed due to infrastructure errors in {handle.id} "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        try:
                            await self.sandbox_pool.remove(handle)
                        except Exception:
                            pass
                        failed_attempts_time += time.perf_counter() - attempt_start

                        if attempt == max_retries - 1:
                            logger.error(
                                f"[{sample.metadata.get('subset_idx')}] All {max_retries} "
                                f"sandbox attempts failed - giving up"
                            )
                            I3CodeEnvironment._reward_count += 1
                            I3CodeEnvironment._reward_errors += 1
                            self._log_progress()
                            return _error_result(
                                acquire_time=cumulative_acquire_time,
                                retry_count=attempt + 1,
                                test_run=test_run,
                                failed_time=failed_attempts_time,
                                error_msg="All tests failed (infrastructure errors)",
                            )
                        continue

                    # Compute pass rate
                    results = test_run.results
                    pass_rate = sum(results) / len(results)
                    passed = pass_rate == 1.0

                    I3CodeEnvironment._reward_count += 1
                    if passed:
                        I3CodeEnvironment._reward_passed += 1

                    total_reward_time = time.perf_counter() - reward_start

                    logger.debug(
                        f"[{sample.metadata.get('subset_idx')}] Tests complete: "
                        f"{sum(results)}/{len(results)} passed (pass_rate={pass_rate:.2%}) | "
                        f"Acquire={cumulative_acquire_time:.1f}s, Tests={test_time:.1f}s, "
                        f"Total={total_reward_time:.1f}s"
                    )
                    self._log_progress()

                    # Release sandbox back to pool
                    await self.sandbox_pool.release(handle)

                    # Build comprehensive sample_metrics for debugging
                    sample_metrics: dict[str, float] = {
                        "passed": 1.0 if passed else 0.0,
                        "pass_rate": pass_rate,
                        "num_test_cases": float(len(results)),
                        "sandbox_error": 0.0,
                        "ops_semaphore_wait_time": cumulative_semaphore_wait_time,
                        "sandbox_acquire_time": cumulative_acquire_time,
                        "test_time": test_time,
                        "retry_count": float(attempt),
                        "failed_attempts_time": failed_attempts_time,
                        "total_reward_time": total_reward_time,
                        "generated_code_chars": code_chars,
                        "num_timeouts": float(test_run.num_timeouts),
                        "num_errors": float(test_run.num_errors),
                    }
                    # Add all timing breakdown from test run
                    sample_metrics.update(test_run.timings)

                    return RewardResult(
                        total_reward=1.0 if passed else 0.0,
                        sample_metrics=sample_metrics,
                        info_turns=self._build_info_turns(
                            generated_code, test_run=test_run,
                        ),
                    )

                except (SandboxNotRunningError, SandboxError) as e:
                    error_msg = str(e)[:200]
                    logger.warning(
                        f"Sandbox error in {handle.id} "
                        f"(attempt {attempt + 1}/{max_retries}): {error_msg}"
                    )
                    try:
                        await self.sandbox_pool.remove(handle)
                    except Exception:
                        pass
                    failed_attempts_time += time.perf_counter() - attempt_start

                    if attempt == max_retries - 1:
                        logger.error(
                            f"[{sample.metadata.get('subset_idx')}] All {max_retries} "
                            f"sandbox attempts failed - giving up"
                        )
                        I3CodeEnvironment._reward_count += 1
                        I3CodeEnvironment._reward_errors += 1
                        self._log_progress()
                        return _error_result(
                            acquire_time=cumulative_acquire_time,
                            retry_count=attempt + 1,
                            failed_time=failed_attempts_time,
                            error_msg=f"SandboxError: {error_msg}",
                        )
                    continue

                except Exception as e:
                    error_msg = str(e)[:200]
                    logger.error(
                        f"[{sample.metadata.get('subset_idx')}] Error in {handle.id}: "
                        f"{error_msg}"
                    )
                    # Release sandbox on non-infrastructure errors
                    try:
                        await self.sandbox_pool.release(handle)
                    except Exception:
                        pass
                    I3CodeEnvironment._reward_count += 1
                    I3CodeEnvironment._reward_errors += 1
                    self._log_progress()
                    return _error_result(
                        acquire_time=cumulative_acquire_time,
                        retry_count=attempt,
                        failed_time=failed_attempts_time,
                        error_msg=f"Unexpected error: {error_msg}",
                    )

            except Exception as e:
                error_msg = str(e)[:200]
                logger.warning(
                    f"[{sample.metadata.get('subset_idx')}] Error acquiring sandbox "
                    f"(attempt {attempt + 1}/{max_retries}): {error_msg}"
                )
                failed_attempts_time += time.perf_counter() - attempt_start
                if attempt == max_retries - 1:
                    logger.error(
                        f"[{sample.metadata.get('subset_idx')}] Failed to acquire sandbox "
                        f"after {max_retries} attempts - giving up"
                    )
                    I3CodeEnvironment._reward_count += 1
                    I3CodeEnvironment._reward_errors += 1
                    self._log_progress()
                    return _error_result(
                        acquire_time=cumulative_acquire_time,
                        retry_count=attempt + 1,
                        failed_time=failed_attempts_time,
                        error_msg=f"Failed to acquire sandbox: {error_msg}",
                    )
                continue

        # Should not reach here, but just in case
        I3CodeEnvironment._reward_count += 1
        I3CodeEnvironment._reward_errors += 1
        self._log_progress()
        return _error_result(
            acquire_time=cumulative_acquire_time,
            retry_count=max_retries,
            failed_time=failed_attempts_time,
            error_msg="Exhausted all retries",
        )

    # ── Progress logging ─────────────────────────────────────────────

    def _log_progress(self) -> None:
        """Log INFO-level progress summary periodically."""
        count = I3CodeEnvironment._reward_count
        # Always log the first completion, then every N completions
        if count == 1 or count % self._progress_log_interval == 0:
            passed = I3CodeEnvironment._reward_passed
            no_code = I3CodeEnvironment._reward_no_code
            errors = I3CodeEnvironment._reward_errors
            tested = count - no_code - errors
            pass_pct = (passed / tested * 100) if tested > 0 else 0
            pool_ready = self.sandbox_pool.ready_queue.qsize()
            pool_in_use = len(self.sandbox_pool.in_use_sandboxes)
            logger.info(
                f"[{self.name}] {count} completions scored | "
                f"{passed}/{tested} passed ({pass_pct:.0f}%) | "
                f"{no_code} no-code, {errors} errors | "
                f"pool: {pool_ready} ready, {pool_in_use} in-use"
            )

    # ── Cleanup ────────────────────────────────────────────────────────

    def _cleanup_sandboxes(self) -> None:
        """Cleanup sandboxes synchronously on exit."""
        try:
            # Try to get event loop and run async shutdown
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.run_until_complete(self.sandbox_pool.shutdown())
                    return
            except RuntimeError:
                # No event loop available, fall through to sync cleanup
                pass

            # Fallback: sync cleanup
            self.sandbox_pool.shutdown_sync()

        except Exception as e:
            logger.error(f"Error during cleanup: {repr(e)}")

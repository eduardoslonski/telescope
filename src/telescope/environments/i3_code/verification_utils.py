
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from telescope.environments._sandbox import (
    SandboxCommandTimeoutError,
    SandboxHandle,
    SandboxProvider,
)

from .deepcoder_utils import (
    BASE_IMPORTS,
    generate_cb_wrapper_script,
    process_input_output,
)
from .sandbox_utils import upload_and_extract_bundle

logger = logging.getLogger(__name__)


@dataclass
class TestRunResult:
    """Rich result from run_test_cases with timing breakdown for debugging."""

    results: list[bool]  # Only non-None results (infrastructure errors filtered out)
    raw_results: list[bool | None]  # All results including None (infrastructure errors)
    timings: dict[str, float] = field(default_factory=dict)
    # Breakdown of timings (all in seconds):
    #   bundle_build_time, upload_semaphore_wait_time, bundle_upload_time,
    #   bundle_extract_time, bundle_total_time, bundle_size_bytes,
    #   tests_execution_time (wall time for all tests to finish),
    #   slowest_test_time, fastest_test_time, median_test_time,
    num_timeouts: int = 0  # Tests that hit timeout_per_test
    num_errors: int = 0  # Tests with infrastructure errors (None results)
    test_type: str = ""  # "stdin" or "func_call"

    # Per-test detail from in-sandbox runner (all tests, passed and failed)
    # Each entry: {"i": int, "passed": bool, "input": str, "stdout": str,
    #              "stderr": str, "expected": str, "exit_code": int,
    #              "actual_result": str, "error": str, "timeout": bool, ...}
    test_details: list[dict] = field(default_factory=list)
    # Runner-level stderr (e.g. import errors, runner crashes)
    runner_stderr: str = ""


# ---------------------------------------------------------------------------
# In-sandbox test runner script
# ---------------------------------------------------------------------------
# This Python source is uploaded into the sandbox as runner.py and executed
# with a single execute_command call.  It runs every test case as a separate
# subprocess (with per-test timeout + memory limit via ulimit), compares
# results, and prints a single JSON blob to stdout.
#
# The ``workers`` count is read from config.json so the host can control
# parallelism per provider (e.g. 32 for Prime/Daytona, 1 for Modal).
# ---------------------------------------------------------------------------

_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""In-sandbox test runner — executes all tests, outputs JSON results."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation

# Max chars for captured stdout/stderr per test (keep JSON output manageable)
_MAX_OUTPUT_CHARS = 1000

def _trunc(s, limit=_MAX_OUTPUT_CHARS):
    if len(s) <= limit:
        return s
    return s[:limit] + "... [truncated]"

# ── stdout comparison (mirrors host-side compare_stdout_results) ──────────

def _split_lines(s):
    return [l.strip() for l in s.strip().splitlines() if l.strip()]

def _tokenise(s):
    return [l.split() for l in _split_lines(s)]

def _flatten(tok):
    return [t for line in tok for t in line]

def compare_stdout(actual, expected, tolerance=1e-3):
    # 1. Trimmed string
    if actual.strip() == expected.strip():
        return True
    # 2. Line-wise
    if _split_lines(actual) == _split_lines(expected):
        return True
    # 3. Token-wise
    if _tokenise(actual) == _tokenise(expected):
        return True
    # 4. Numeric tolerance
    try:
        ta, tb = _flatten(_tokenise(actual)), _flatten(_tokenise(expected))
        if len(ta) == len(tb) and ta:
            da = [Decimal(t) for t in ta]
            db = [Decimal(t) for t in tb]
            tol = Decimal(str(tolerance))
            if all(abs(a - b) <= tol for a, b in zip(da, db)):
                return True
    except Exception:
        pass
    return False

# ── func_call comparison (mirrors host-side logic) ────────────────────────

def compare_func(exec_out, expected):
    if isinstance(exec_out, tuple):
        exec_out = list(exec_out)
    ok = (exec_out == expected)
    if isinstance(expected, list):
        ok = ok or (exec_out == expected[0])
    try:
        if isinstance(exec_out[0], tuple):
            conv = [list(x) for x in exec_out]
            ok = ok or (conv == expected[0])
    except Exception:
        pass
    return bool(ok)

# ── single test execution ─────────────────────────────────────────────────

def run_one(i, mode, base, timeout):
    t0 = time.monotonic()
    try:
        if mode == "stdin":
            script = os.path.join(base, "script.py")
            inp_path = os.path.join(base, "inputs", str(i) + ".in")
            with open(inp_path) as f:
                input_data = f.read()
            with open(inp_path) as f:
                proc = subprocess.run(
                    ["bash", "-c", "ulimit -v 10485760; python " + script],
                    stdin=f, capture_output=True, text=True, timeout=timeout,
                )
            elapsed = time.monotonic() - t0
            with open(os.path.join(base, "expected", str(i) + ".out")) as f:
                expected = f.read()
            if proc.returncode != 0:
                return {"i": i, "passed": False, "time": elapsed,
                        "exit_code": proc.returncode,
                        "input": _trunc(input_data),
                        "stdout": _trunc(proc.stdout),
                        "expected": _trunc(expected),
                        "stderr": _trunc(proc.stderr)}
            passed = compare_stdout(proc.stdout, expected)
            return {"i": i, "passed": passed, "time": elapsed,
                    "exit_code": proc.returncode,
                    "input": _trunc(input_data),
                    "stdout": _trunc(proc.stdout),
                    "expected": _trunc(expected),
                    "stderr": _trunc(proc.stderr)}
        else:
            script = os.path.join(base, "scripts", "script_" + str(i) + ".py")
            # Read the wrapper script to extract input info
            with open(script) as f:
                script_src = f.read()
            proc = subprocess.run(
                ["bash", "-c", "ulimit -v 10485760; python " + script],
                capture_output=True, text=True, timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            with open(os.path.join(base, "expected", str(i) + ".json")) as f:
                expected = json.load(f)
            if proc.returncode != 0:
                return {"i": i, "passed": False, "time": elapsed,
                        "exit_code": proc.returncode,
                        "input": _trunc(script_src),
                        "stdout": _trunc(proc.stdout),
                        "expected": _trunc(str(expected)),
                        "stderr": _trunc(proc.stderr)}
            try:
                data = json.loads(proc.stdout.strip())
                if not data.get("success"):
                    return {"i": i, "passed": False, "time": elapsed,
                            "exit_code": proc.returncode,
                            "input": _trunc(script_src),
                            "stdout": _trunc(proc.stdout),
                            "expected": _trunc(str(expected)),
                            "stderr": _trunc(proc.stderr)}
                passed = compare_func(data["result"], expected)
                return {"i": i, "passed": passed, "time": elapsed,
                        "exit_code": proc.returncode,
                        "input": _trunc(script_src),
                        "actual_result": _trunc(str(data["result"])),
                        "expected": _trunc(str(expected)),
                        "stderr": _trunc(proc.stderr)}
            except Exception as ex:
                return {"i": i, "passed": False, "time": elapsed,
                        "exit_code": proc.returncode,
                        "input": _trunc(script_src),
                        "stdout": _trunc(proc.stdout),
                        "expected": _trunc(str(expected)),
                        "stderr": _trunc(proc.stderr),
                        "error": str(ex)[:200]}
    except subprocess.TimeoutExpired:
        # Try to read input for context even on timeout
        inp_str = ""
        try:
            if mode == "stdin":
                with open(os.path.join(base, "inputs", str(i) + ".in")) as f:
                    inp_str = _trunc(f.read())
            else:
                with open(os.path.join(base, "scripts", "script_" + str(i) + ".py")) as f:
                    inp_str = _trunc(f.read())
        except Exception:
            pass
        return {"i": i, "passed": False, "time": float(timeout), "timeout": True,
                "input": inp_str}
    except Exception as e:
        return {"i": i, "passed": False, "time": time.monotonic() - t0, "error": str(e)[:200]}

# ── main ──────────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "config.json")) as f:
        cfg = json.load(f)
    mode, timeout, n = cfg["mode"], cfg["timeout"], cfg["num_tests"]
    # Workers controlled by host — allows per-provider tuning
    max_workers = cfg.get("workers", 32)
    results = [None] * n
    t0 = time.monotonic()
    workers = min(n, max_workers)
    if workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_one, i, mode, base, timeout): i for i in range(n)}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = {"i": idx, "passed": False, "time": 0, "error": str(e)[:200]}
    print(json.dumps({"results": results, "total_time": time.monotonic() - t0}))

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Common helper: upload bundle with runner, execute, parse JSON results
# ---------------------------------------------------------------------------

@dataclass
class _RunResult:
    """Internal result from _upload_and_run with all extracted data."""
    raw_results: list[bool | None]
    timings: dict[str, float]
    test_details: list[dict]  # Per-test detail dicts from runner
    runner_stderr: str  # Runner-level stderr


async def _upload_and_run(
    provider: SandboxProvider,
    handle: SandboxHandle,
    file_map: dict[str, str],
    num_tests: int,
    timeout_per_test: int,
    workspace_path: str = "/sandbox-workspace",
) -> _RunResult:
    """Upload a bundle containing the runner + test files, execute the runner
    with a single ``execute`` call, and parse the structured JSON output.

    Returns a ``_RunResult`` with raw pass/fail booleans, timings,
    per-test detail dicts (stdout/stderr for failed tests), and runner stderr.
    """
    bundle_id = uuid.uuid4().hex
    archive_remote = f"{workspace_path}/bundle_{bundle_id}.tar.gz"
    extract_dir = f"{workspace_path}/bundle_{bundle_id}"

    logger.debug(
        f"[runner] Bundling for sandbox {handle.id}: "
        f"archive={archive_remote}, extract_dir={extract_dir}, "
        f"files_in_bundle={len(file_map)}"
    )

    # 1. Upload & extract bundle (1 upload + 1 extract command)
    bundle_timings = await upload_and_extract_bundle(
        provider=provider,
        handle=handle,
        file_map=file_map,
        archive_remote=archive_remote,
        extract_dir=extract_dir,
    )

    # 2. Execute runner — single command for ALL tests
    # Timeout: generous enough for sequential worst-case, but the runner
    # executes tests in parallel so it'll normally finish much faster.
    # The outer asyncio.wait_for(180s) in environment.py is the hard cap.
    runner_timeout = max(timeout_per_test * num_tests + 30, 120)
    runner_cmd = f"python {extract_dir}/runner.py"

    tests_start = time.perf_counter()
    try:
        resp = await provider.execute(handle, runner_cmd, timeout=runner_timeout)
    except SandboxCommandTimeoutError:
        tests_wall_time = time.perf_counter() - tests_start
        logger.warning(
            f"Runner timed out after {runner_timeout}s in sandbox {handle.id}"
        )
        return _RunResult(
            raw_results=[None] * num_tests,
            timings={
                **bundle_timings,
                "tests_execution_time": tests_wall_time,
                "num_timeouts": 0.0,
                "num_errors": float(num_tests),
            },
            test_details=[],
            runner_stderr="[runner timed out]",
        )
    except Exception:
        # SandboxNotRunningError, etc. — let environment.py handle
        raise

    tests_wall_time = time.perf_counter() - tests_start
    runner_stderr = (resp.stderr or "").strip()

    # 3. Parse runner output
    if resp.exit_code != 0:
        logger.warning(
            f"Runner exited with code {resp.exit_code} in sandbox {handle.id}. "
            f"stderr: {runner_stderr[:300]}"
        )
        return _RunResult(
            raw_results=[None] * num_tests,
            timings={
                **bundle_timings,
                "tests_execution_time": tests_wall_time,
                "num_timeouts": 0.0,
                "num_errors": float(num_tests),
            },
            test_details=[],
            runner_stderr=runner_stderr[:2000],
        )

    try:
        output = json.loads(resp.stdout.strip())
        runner_results = output["results"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(
            f"Failed to parse runner JSON in sandbox {handle.id}: {str(e)[:200]}. "
            f"stdout[:500]: {(resp.stdout or '')[:500]}"
        )
        return _RunResult(
            raw_results=[None] * num_tests,
            timings={
                **bundle_timings,
                "tests_execution_time": tests_wall_time,
                "num_timeouts": 0.0,
                "num_errors": float(num_tests),
            },
            test_details=[],
            runner_stderr=runner_stderr[:2000],
        )

    # 4. Map runner results → (raw_results, timings, test_details)
    raw_results: list[bool | None] = []
    per_test_times: list[float] = []
    test_details: list[dict] = []
    num_timeouts = 0
    num_errors = 0

    for r in runner_results:
        if r is None:
            raw_results.append(None)
            per_test_times.append(0.0)
            num_errors += 1
            continue
        per_test_times.append(r.get("time", 0.0))
        if r.get("error"):
            raw_results.append(None)
            num_errors += 1
        elif r.get("timeout"):
            raw_results.append(False)
            num_timeouts += 1
        else:
            raw_results.append(bool(r.get("passed", False)))
        # Keep detail for all tests (passed and failed)
        test_details.append(r)

    timings: dict[str, float] = {
        **bundle_timings,
        "tests_execution_time": tests_wall_time,
        "num_timeouts": float(num_timeouts),
        "num_errors": float(num_errors),
    }
    if per_test_times:
        sorted_times = sorted(per_test_times)
        timings["slowest_test_time"] = sorted_times[-1]
        timings["fastest_test_time"] = sorted_times[0]
        mid = len(sorted_times) // 2
        timings["median_test_time"] = (
            sorted_times[mid]
            if len(sorted_times) % 2 == 1
            else (sorted_times[mid - 1] + sorted_times[mid]) / 2.0
        )

    return _RunResult(
        raw_results=raw_results,
        timings=timings,
        test_details=test_details,
        runner_stderr=runner_stderr[:2000] if runner_stderr else "",
    )


# ---------------------------------------------------------------------------
# Public API (unified — works with any SandboxProvider)
# ---------------------------------------------------------------------------

async def run_standard_input(
    generated_code: str,
    inputs: list,
    outputs: list,
    provider: SandboxProvider,
    handle: SandboxHandle,
    timeout_per_test: int,
    num_workers: int = 32,
    workspace_path: str = "/sandbox-workspace",
) -> _RunResult:
    """Runs stdin/stdout test cases using the in-sandbox runner.

    Returns:
        _RunResult with raw pass/fail, timings, per-test details, runner stderr.
    """
    file_map: dict[str, str] = {
        "runner.py": _RUNNER_SCRIPT,
        "script.py": generated_code,
    }

    # Input files
    for idx, test_input in enumerate(inputs):
        if isinstance(test_input, list):
            test_input = "\n".join(str(k) for k in test_input)
        file_map[f"inputs/{idx}.in"] = str(test_input)

    # Expected output files (pre-processed for direct comparison)
    for idx, test_output in enumerate(outputs):
        if isinstance(test_output, list):
            test_output = "\n".join(str(k) for k in test_output)
        file_map[f"expected/{idx}.out"] = str(test_output)

    # Config for the runner
    file_map["config.json"] = json.dumps({
        "mode": "stdin",
        "timeout": timeout_per_test,
        "num_tests": len(inputs),
        "workers": num_workers,
    })

    logger.debug(
        f"[stdin] Runner bundle for sandbox {handle.id}: "
        f"{len(inputs)} tests, timeout={timeout_per_test}s/test, "
        f"workers={num_workers}"
    )

    return await _upload_and_run(
        provider, handle, file_map, len(inputs), timeout_per_test,
        workspace_path=workspace_path,
    )


async def run_func_call(
    generated_code: str,
    fn_name: str,
    inputs: list,
    outputs: list,
    provider: SandboxProvider,
    handle: SandboxHandle,
    timeout_per_test: int,
    num_workers: int = 32,
    workspace_path: str = "/sandbox-workspace",
) -> _RunResult:
    """Runs function-based test cases using the in-sandbox runner.

    Returns:
        _RunResult with raw pass/fail, timings, per-test details, runner stderr.
    """
    file_map: dict[str, str] = {
        "runner.py": _RUNNER_SCRIPT,
    }

    # Wrapper scripts (one per test, each calls the function with its inputs)
    for i, test_input in enumerate(inputs):
        script = generate_cb_wrapper_script(generated_code, fn_name, test_input)
        file_map[f"scripts/script_{i}.py"] = script

    # Expected output files
    # outputs[i] may be a string (double-encoded JSON) or a Python object.
    # We do the final parse host-side and store as proper JSON so the runner
    # just needs json.load().
    for idx, test_output in enumerate(outputs):
        if isinstance(test_output, str):
            try:
                final_expected = json.loads(test_output)
            except (json.JSONDecodeError, TypeError):
                final_expected = test_output
        else:
            final_expected = test_output
        file_map[f"expected/{idx}.json"] = json.dumps(final_expected)

    # Config for the runner
    file_map["config.json"] = json.dumps({
        "mode": "func_call",
        "timeout": timeout_per_test,
        "num_tests": len(inputs),
        "workers": num_workers,
    })

    logger.debug(
        f"[func] Runner bundle for sandbox {handle.id}: "
        f"{len(inputs)} tests, fn={fn_name}, timeout={timeout_per_test}s/test, "
        f"workers={num_workers}"
    )

    return await _upload_and_run(
        provider, handle, file_map, len(inputs), timeout_per_test,
        workspace_path=workspace_path,
    )


async def run_test_cases(
    generated_code: str,
    verification_info: dict,
    provider: SandboxProvider,
    handle: SandboxHandle,
    num_workers: int = 32,
    workspace_path: str = "/sandbox-workspace",
) -> TestRunResult:
    """Run all test cases and return rich result with timing breakdown.

    Returns:
        TestRunResult with filtered results, raw results, timings, error counts,
        per-test details, and runner stderr.
    """
    generated_code = f"{BASE_IMPORTS}\n{generated_code}"
    inputs = []
    outputs = []
    for test_case_inputs, test_case_outputs in zip(
        verification_info["test_case_inputs"], verification_info["test_case_outputs"]
    ):
        # deserialize the input and output
        test_case_inputs = json.loads(test_case_inputs)
        test_case_outputs = json.loads(test_case_outputs)
        test_case_inputs, test_case_outputs = process_input_output(test_case_inputs, test_case_outputs)
        inputs.append(test_case_inputs)
        outputs.append(test_case_outputs)

    if not verification_info["fn_name"]:
        run_result = await run_standard_input(
            generated_code,
            inputs=inputs,
            outputs=outputs,
            provider=provider,
            handle=handle,
            timeout_per_test=verification_info["timeout"],
            num_workers=num_workers,
            workspace_path=workspace_path,
        )
        test_type = "stdin"
    else:
        run_result = await run_func_call(
            generated_code,
            fn_name=verification_info["fn_name"],
            inputs=inputs,
            outputs=outputs,
            provider=provider,
            handle=handle,
            timeout_per_test=verification_info["timeout"],
            num_workers=num_workers,
            workspace_path=workspace_path,
        )
        test_type = "func_call"

    filtered_results = [r for r in run_result.raw_results if r is not None]

    return TestRunResult(
        results=filtered_results,
        raw_results=run_result.raw_results,
        timings=run_result.timings,
        num_timeouts=int(run_result.timings.get("num_timeouts", 0)),
        num_errors=int(run_result.timings.get("num_errors", 0)),
        test_type=test_type,
        test_details=run_result.test_details,
        runner_stderr=run_result.runner_stderr,
    )

"""
Subprocess-isolated environment worker pool for compute_reward.

Runs reward computation in a separate process (own GIL, crash isolation)
so CPU-heavy reward logic doesn't block the orchestrator's async event loop
or contend with the HTTP thread pool.

Each worker process loads its own copy of all environments on startup.
The orchestrator sends (env_name, state_dict, eos_token) and receives
a RewardResult back.  Communication uses ProcessPoolExecutor with pickle
serialization.

Note: env_response is NOT isolated here because:
1. It mutates state on every turn → expensive per-turn serialization
2. Most env_response calls are fast (game logic, text parsing)
3. When slow (sandbox code execution), the bottleneck is the external
   call (Modal API), not local CPU / GIL contention
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

from telescope.utils.tlog import get_logger

_log = get_logger("orchestrator")

# ── Worker-side globals (one set per subprocess) ──────────────────────

_worker_envs: dict[str, Any] = {}
_worker_loop: asyncio.AbstractEventLoop | None = None


def _init_worker(env_names: list[str], env_kwargs_by_name: dict[str, dict]):
    """Initialize environment instances inside the worker process."""
    import asyncio as _aio
    global _worker_envs, _worker_loop

    from telescope.environments.registry import get_environment

    _worker_loop = _aio.new_event_loop()
    _aio.set_event_loop(_worker_loop)

    for name in env_names:
        kwargs = env_kwargs_by_name.get(name, {})
        try:
            env = get_environment(name, **kwargs)
            _worker_envs[name] = env
        except Exception as e:
            # Log but don't crash — the worker can still serve other envs
            import traceback
            traceback.print_exc()
            print(f"[EnvWorker] Failed to load environment '{name}': {e}")


def _run_compute_reward(
    env_name: str,
    is_multi_turn: bool,
    completion_or_state: Any,
    sample_dict: dict,
    eos_token: str,
) -> dict:
    """Execute compute_reward in the worker process.

    Returns a dict representation of RewardResult so it doesn't require
    importing the dataclass in the parent process for unpickling.
    """
    from telescope.environments.base import Sample, RolloutState, TrajectoryStep

    env = _worker_envs.get(env_name)
    if env is None:
        return {
            "total_reward": 0.0,
            "sample_metrics": {},
            "golden_answers": {},
            "info_turns": [],
            "sample_tags": {},
            "_error": f"Environment '{env_name}' not loaded in worker",
        }

    sample = Sample(**sample_dict)

    try:
        if is_multi_turn:
            # Reconstruct RolloutState from serialized dict
            state_dict = completion_or_state
            trajectory = [
                TrajectoryStep(**step_dict)
                for step_dict in state_dict.pop("trajectory", [])
            ]
            state = RolloutState(
                sample=sample,
                env_name=env_name,
                trajectory=trajectory,
                is_completed=state_dict.get("is_completed", False),
                stop_reason=state_dict.get("stop_reason"),
                error=state_dict.get("error"),
                custom=state_dict.get("custom", {}),
            )
            coro = env.compute_reward(state, eos_token)
        else:
            completion_text = completion_or_state
            coro = env.compute_reward(completion_text, sample, eos_token)

        result = _worker_loop.run_until_complete(coro)
        return {
            "total_reward": result.total_reward,
            "sample_metrics": dict(result.sample_metrics),
            "golden_answers": dict(result.golden_answers),
            "info_turns": list(result.info_turns),
            "sample_tags": dict(result.sample_tags),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "total_reward": 0.0,
            "sample_metrics": {},
            "golden_answers": {},
            "info_turns": [],
            "sample_tags": {},
            "_error": str(e),
        }


# ── Orchestrator-side pool ────────────────────────────────────────────

class EnvWorkerPool:
    """
    Process pool for isolated reward computation.

    Usage::

        pool = EnvWorkerPool(
            env_names=["countdown", "wordle"],
            env_kwargs_by_name={"countdown": {"target": 24}},
            num_workers=4,
        )

        # Single-turn reward
        result = await pool.compute_reward(
            env_name="countdown",
            is_multi_turn=False,
            completion_or_state="The answer is 24",
            sample=sample,
            eos_token="<eos>",
        )

        # Multi-turn reward
        result = await pool.compute_reward(
            env_name="wordle",
            is_multi_turn=True,
            completion_or_state=state,   # RolloutState
            sample=state.sample,
            eos_token="<eos>",
        )
    """

    def __init__(
        self,
        env_names: list[str],
        env_kwargs_by_name: dict[str, dict] | None = None,
        num_workers: int = 4,
    ):
        self._env_names = env_names
        self._env_kwargs = env_kwargs_by_name or {}
        self._num_workers = num_workers
        self._executor: ProcessPoolExecutor | None = None

    def start(self):
        """Create the process pool and initialize workers."""
        self._executor = ProcessPoolExecutor(
            max_workers=self._num_workers,
            mp_context=mp.get_context("forkserver"),
            initializer=_init_worker,
            initargs=(self._env_names, self._env_kwargs),
        )
        _log.info(
            f"EnvWorkerPool started with {self._num_workers} workers "
            f"(envs: {self._env_names})"
        )

    def shutdown(self):
        """Shutdown the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            _log.info("EnvWorkerPool shut down")

    async def compute_reward(
        self,
        env_name: str,
        is_multi_turn: bool,
        completion_or_state: Any,
        sample: Any,
        eos_token: str,
    ):
        """
        Run compute_reward in a worker process.

        Args:
            env_name: Name of the environment
            is_multi_turn: Whether this is a multi-turn rollout
            completion_or_state: For single-turn: completion text (str).
                For multi-turn: RolloutState object (will be serialized).
            sample: Sample object
            eos_token: EOS token string

        Returns:
            RewardResult dataclass
        """
        from telescope.environments.base import RewardResult, RolloutState

        # Serialize sample for pickling
        sample_dict = {
            "prompt": sample.prompt,
            "answer": sample.answer,
            "metadata": sample.metadata,
        }

        # Serialize state for multi-turn
        if is_multi_turn and isinstance(completion_or_state, RolloutState):
            state = completion_or_state
            serialized = {
                "trajectory": [
                    {
                        "prompt": step.prompt,
                        "completion": step.completion,
                        "prompt_token_ids": step.prompt_token_ids,
                        "completion_token_ids": step.completion_token_ids,
                        "completion_logprobs": step.completion_logprobs,
                        "is_truncated": step.is_truncated,
                    }
                    for step in state.trajectory
                ],
                "is_completed": state.is_completed,
                "stop_reason": state.stop_reason,
                "error": state.error,
                "custom": state.custom,
            }
            completion_or_state = serialized

        loop = asyncio.get_event_loop()
        result_dict = await loop.run_in_executor(
            self._executor,
            _run_compute_reward,
            env_name,
            is_multi_turn,
            completion_or_state,
            sample_dict,
            eos_token,
        )

        if "_error" in result_dict:
            _log.warning(
                f"EnvWorkerPool compute_reward error for {env_name}: "
                f"{result_dict['_error']}"
            )

        return RewardResult(
            total_reward=result_dict["total_reward"],
            sample_metrics=result_dict.get("sample_metrics", {}),
            golden_answers=result_dict.get("golden_answers", {}),
            info_turns=result_dict.get("info_turns", []),
            sample_tags=result_dict.get("sample_tags", {}),
        )

"""
Composable reward system for telescope environments.

Provides a ``Rubric`` that composes multiple reward functions — sync or async,
weighted or metric-only — into a single ``RewardResult``.

Example usage::

    from telescope.environments.rewards import Rubric

    rubric = Rubric()
    rubric.add_reward(format_reward, range_min=0, range_max=1)
    rubric.add_reward(equation_reward, range_min=0, range_max=1)

    async def compute_reward(self, completion, sample, eos_token=""):
        return await rubric.score(completion=completion, sample=sample)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from telescope.environments.base import RewardResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async utility
# ---------------------------------------------------------------------------

async def maybe_await(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call *func* and ``await`` the result if it is a coroutine."""
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


# ---------------------------------------------------------------------------
# Signature introspection
# ---------------------------------------------------------------------------

def _build_call_kwargs(
    func: Callable, available: dict[str, Any]
) -> dict[str, Any]:
    """Return only the *available* kwargs that *func* declares in its signature.

    If *func* accepts ``**kwargs``, all non-None entries are forwarded.
    Otherwise, only explicitly named parameters are included.
    """
    sig = inspect.signature(func)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        return {k: v for k, v in available.items() if v is not None}
    return {
        name: available[name]
        for name in sig.parameters
        if name in available and available[name] is not None
    }


# ---------------------------------------------------------------------------
# Reward entry
# ---------------------------------------------------------------------------

@dataclass
class _RewardEntry:
    func: Callable
    weight: float
    name: str
    golden_answer: str | None
    range_min: float | None
    range_max: float | None
    invert: bool


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

class Rubric:
    """Compose multiple reward functions into a single :class:`RewardResult`.

    Reward functions may be sync or async — the rubric uses :func:`maybe_await`
    to handle both transparently.  Each function is called with only the keyword
    arguments it declares (via signature introspection).

    Functions return either ``float`` or ``tuple[float, str | None]`` where the
    second element is a golden-answer string.
    """

    def __init__(self) -> None:
        self._entries: list[_RewardEntry] = []

    # -- registration -------------------------------------------------------

    def add_reward(
        self,
        func: Callable,
        weight: float = 1.0,
        *,
        name: str | None = None,
        golden_answer: str | None = None,
        range_min: float | None = None,
        range_max: float | None = None,
        invert: bool = False,
    ) -> "Rubric":
        """Register a reward function with a weight.

        Returns *self* for chaining.
        """
        self._entries.append(
            _RewardEntry(
                func=func,
                weight=weight,
                name=name or func.__name__,
                golden_answer=golden_answer,
                range_min=range_min,
                range_max=range_max,
                invert=invert,
            )
        )
        return self

    def add_metric(
        self,
        func: Callable,
        *,
        name: str | None = None,
        golden_answer: str | None = None,
        range_min: float | None = None,
        range_max: float | None = None,
        invert: bool = False,
    ) -> "Rubric":
        """Register a metric (weight = 0). Tracked but does not affect ``total_reward``."""
        return self.add_reward(
            func,
            weight=0.0,
            name=name,
            golden_answer=golden_answer,
            range_min=range_min,
            range_max=range_max,
            invert=invert,
        )

    # -- metrics_ranges property -------------------------------------------

    @property
    def metrics_ranges(self) -> dict[str, dict]:
        """Auto-generate ``metrics_ranges`` from registered entries."""
        ranges: dict[str, dict] = {}
        for entry in self._entries:
            d: dict[str, Any] = {}
            if entry.range_min is not None:
                d["min"] = entry.range_min
            if entry.range_max is not None:
                d["max"] = entry.range_max
            if entry.invert:
                d["invert"] = True
            if d:
                ranges[entry.name] = d
        return ranges

    # -- scoring ------------------------------------------------------------

    async def score(
        self,
        *,
        completion: str | None = None,
        sample: Any | None = None,
        state: Any | None = None,
        eos_token: str = "",
        extra_sample_metrics: dict[str, float] | None = None,
        extra_golden_answers: dict[str, str | None] | None = None,
        extra_info_turns: list[dict[str, Any]] | None = None,
        extra_sample_tags: dict[str, str] | None = None,
    ) -> RewardResult:
        """Call every registered function, compute weighted total, return :class:`RewardResult`."""

        available: dict[str, Any] = {
            "completion": completion,
            "sample": sample,
            "state": state,
            "eos_token": eos_token,
        }

        total_reward = 0.0
        sample_metrics: dict[str, float] = {}
        golden_answers: dict[str, str | None] = {}

        for entry in self._entries:
            call_kwargs = _build_call_kwargs(entry.func, available)

            try:
                raw = await maybe_await(entry.func, **call_kwargs)
            except Exception as e:
                logger.warning(
                    "Reward function '%s' raised %s: %s; using 0.0",
                    entry.name,
                    type(e).__name__,
                    e,
                )
                raw = 0.0

            # Unpack (score, golden_answer) or just score
            if isinstance(raw, tuple):
                score_val = float(raw[0])
                ga = raw[1] if len(raw) > 1 else None
            else:
                score_val = float(raw)
                ga = None

            total_reward += score_val * entry.weight
            sample_metrics[entry.name] = score_val

            # Golden answer: explicit from return > explicit from registration > skip
            if ga is not None:
                golden_answers[entry.name] = ga
            elif entry.golden_answer is not None:
                golden_answers[entry.name] = entry.golden_answer

        # Merge extras
        if extra_sample_metrics:
            sample_metrics.update(extra_sample_metrics)
        if extra_golden_answers:
            golden_answers.update(extra_golden_answers)

        return RewardResult(
            total_reward=total_reward,
            sample_metrics=sample_metrics,
            golden_answers=golden_answers,
            info_turns=extra_info_turns or [],
            sample_tags=extra_sample_tags or {},
        )

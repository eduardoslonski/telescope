"""
DeepDive Environment — Multi-turn web research agent with judge-based reward.

Adapted from prime-envs/deepdive for the telescope framework.

The agent uses search_web, scan_page, and open_lines tools to research
questions on the web, then provides a final answer via the finish tool
(or \\boxed{} when finish_with_tool=False). A judge model evaluates
correctness against a gold answer.

Requires: aiohttp, diskcache, pdfminer.six, openai, httpx
"""

REQUIRED_PACKAGES = ["aiohttp", "diskcache", "pdfminer.six", "openai", "httpx"]

import asyncio
import json
import logging
import os
import re
from typing import Any

import aiohttp
import httpx
from datasets import load_dataset as hf_load_dataset
from openai import AsyncOpenAI

from telescope.environments.base import (
    ChatMessage,
    RewardResult,
    RolloutState,
    Sample,
)
from telescope.environments.rewards import Rubric
from telescope.environments.tool_env import ToolEnvironment

from .config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    METADATA_KEYS,
    PROMPT_SUFFIX,
    SERPER_API_URL,
)
from .formatting import format_search_results, format_serper_results, truncate_text
from .open_one import (
    close_cache,
    close_http_session,
    configure_cache,
    configure_fetch_semaphore,
    configure_http_client,
    configure_thread_pool,
    open_one_result,
)
from .rate_limit import with_rate_limit_retry
from .web_tools import (
    build_explore_block,
    compile_search_pattern,
    normalize_line_ranges,
    render_line_ranges,
    truncate_output,
)

logger = logging.getLogger("deepdive")

# Judge prompt template
JUDGE_PROMPT = """\
You are a helpful judge. Given a question and a reference answer, determine \
if the given response correctly answers the question. The response does not \
need to match the reference answer exactly, but it should contain the correct \
information.

Question: {question}
Reference Answer: {answer}
Response: {completion}

Does the response correctly answer the question? Reply with only "yes" or "no"."""


def _extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...}, handling nested braces."""
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


class DeepDiveEnvironment(ToolEnvironment):
    """
    Multi-turn web research environment with judge-based reward.

    Reward components:
    - judge_reward (weight 1.0): Binary correctness from judge model
    - redundancy_penalty (weight -configurable): Penalises repetitive search queries
    """

    reward_min = 0.0
    reward_max = 1.0

    @property
    def metrics_ranges(self):
        return self.rubric.metrics_ranges

    tags_options: dict[str, list[str]] = {
        "category": [],
        "difficulty": [],
        "source": [],
    }

    def __init__(
        self,
        max_turns: int = 32,
        serper_api_key_var: str = "SERPER_API_KEY",
        judge_api_key_var: str = "OPENAI_API_KEY",
        judge_model: str = "gpt-4.1-mini",
        judge_base_url: str | None = None,
        max_search_results: int = 10,
        max_response_chars: int | float = 20_000,
        serper_timeout: float = 15.0,
        dataset_name: str = DEFAULT_DATASET_NAME,
        dataset_split: str = DEFAULT_DATASET_SPLIT,
        dataset_test_size: float = 0.1,
        dataset_seed: int = 2025,
        redundancy_penalty_weight: float = 0.0,
        finish_with_tool: bool = True,
        tool_call_format: str = "xml",
        log_level: str | int = "INFO",
        # Cache / fetch configuration
        open_max_workers: int = 64,
        open_max_concurrency: int = 64,
        open_max_connections: int = 256,
        open_max_connections_per_host: int = 0,
        cache_dir: str | None = None,
        cache_size_limit_gb: int = 10,
        cache_ttl_seconds: int = 604800,
        cache_shards: int = 8,
        in_memory_cache_max_bytes: int = 16_777_216,
        in_memory_cache_max_entry_bytes: int = 200_000,
        **kwargs,
    ):
        if log_level is not None:
            logger.setLevel(log_level)

        # --- Configuration ---
        self._serper_api_key = os.getenv(serper_api_key_var)
        if not self._serper_api_key:
            raise ValueError(f"Missing Serper API key. Set {serper_api_key_var}.")

        self._judge_model = judge_model
        self._max_search_results = max_search_results
        self._max_response_chars = max(1, int(max_response_chars))
        self._serper_timeout = serper_timeout
        self._dataset_name = dataset_name
        self._dataset_split = dataset_split
        self._dataset_test_size = dataset_test_size
        self._dataset_seed = dataset_seed
        self._redundancy_penalty_weight = redundancy_penalty_weight
        self._finish_with_tool = finish_with_tool

        # --- Open-one infrastructure ---
        configure_thread_pool(max_workers=open_max_workers)
        configure_fetch_semaphore(max_concurrency=open_max_concurrency)
        configure_http_client(
            max_connections=open_max_connections,
            max_connections_per_host=open_max_connections_per_host,
        )
        configure_cache(
            cache_dir=cache_dir,
            size_limit_gb=cache_size_limit_gb,
            ttl_seconds=cache_ttl_seconds,
            cache_shards=cache_shards,
            in_memory_cache_max_bytes=in_memory_cache_max_bytes,
            in_memory_cache_max_entry_bytes=in_memory_cache_max_entry_bytes,
        )

        # --- Judge client ---
        httpx_timeout = httpx.Timeout(1200)
        httpx_limits = httpx.Limits(
            max_connections=8192, max_keepalive_connections=8192
        )
        self._httpx_client = httpx.AsyncClient(
            limits=httpx_limits, timeout=httpx_timeout
        )
        self._judge_client = AsyncOpenAI(
            base_url=judge_base_url,
            api_key=os.getenv(judge_api_key_var) if judge_api_key_var else "EMPTY",
            http_client=self._httpx_client,
        )

        # Rate-limited judge function (created once, reused for all reward calls)
        judge_client = self._judge_client
        judge_model_name = self._judge_model
        concurrency_sem = asyncio.Semaphore(128)
        rate_limit_sem = asyncio.Semaphore(1)
        rate_limit_event = asyncio.Event()
        rate_limit_event.set()

        @with_rate_limit_retry(concurrency_sem, rate_limit_sem, rate_limit_event)
        async def _call_judge(question: str, completion: str, answer: str) -> str:
            response = await judge_client.chat.completions.create(
                model=judge_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=question,
                            answer=answer,
                            completion=completion,
                        ),
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

        self._call_judge = _call_judge

        # Current state reference (set during env_response for finish tool access)
        self._current_state: RolloutState | None = None

        # --- Rubric ---
        self.rubric = Rubric()
        self.rubric.add_reward(self._judge_reward, range_min=0, range_max=1)
        self.rubric.add_reward(
            self._redundancy_penalty_reward,
            weight=-self._redundancy_penalty_weight,
            range_min=0, range_max=1, invert=True,
        )
        self.rubric.add_metric(self._search_web_mean_queries, range_min=0, range_max=10)
        self.rubric.add_metric(self._search_web_error_rate, range_min=0, range_max=1, invert=True)
        self.rubric.add_metric(self._scan_page_error_rate, range_min=0, range_max=1, invert=True)
        self.rubric.add_metric(self._open_lines_error_rate, range_min=0, range_max=1, invert=True)
        if finish_with_tool:
            self.rubric.add_metric(self._finish_error_rate, range_min=0, range_max=1, invert=True)

        # --- Build tools ---
        tools = self._create_tools()

        super().__init__(
            tools=tools,
            max_turns=max_turns,
            tool_call_format=tool_call_format,
            include_tool_schemas_in_prompt=True,
            **kwargs,
        )

    # =========================================================================
    # Tool definitions
    # =========================================================================

    def _create_tools(self) -> list:
        """Create async tool functions as closures capturing *self*."""
        env = self

        async def search_web(
            queries: list[str], num_results_per_query: int = 3
        ) -> str:
            """Search Google with up to 10 queries in parallel. Queries beyond 10 are ignored."""
            if not isinstance(queries, list) or any(
                not isinstance(q, str) for q in queries
            ):
                return "Error: `queries` must be a list of strings."
            queries = [q.strip() for q in queries if q.strip()][:10]
            if not queries:
                return ""

            async def _search_one(query: str) -> str:
                query = query.strip()
                if not query:
                    return ""
                payload = {"q": query}
                headers = {
                    "X-API-KEY": env._serper_api_key,
                    "Content-Type": "application/json",
                }
                timeout = aiohttp.ClientTimeout(total=env._serper_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        SERPER_API_URL, headers=headers, json=payload
                    ) as response:
                        content = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(
                                f"Serper API error {response.status}: {content.strip()}"
                            )
                data = json.loads(content)
                limit = max(
                    1, min(int(num_results_per_query), env._max_search_results)
                )
                formatted = format_serper_results(data, limit, query)
                return truncate_text(formatted, env._max_response_chars)

            results = await asyncio.gather(
                *[_search_one(q) for q in queries]
            )
            return format_search_results(queries, results)

        async def scan_page(
            url: str,
            pattern: str = "",
            context_lines: int = 0,
            max_matches: int = 200,
        ) -> str:
            """Get page metadata and search for specific information. Good to use before open_lines.

            Args:
                url: URL to inspect.
                pattern: Optional regex pattern to match lines (case-insensitive). Empty string means no pattern.
                context_lines: Number of extra lines to include around each match.
                max_matches: Maximum number of matching lines to return.
            """
            result = await open_one_result(url)
            compiled_pattern, pattern_error = compile_search_pattern(
                pattern or None
            )
            context_lines = max(0, int(context_lines))
            max_matches_int = max(0, int(max_matches))
            results_str = build_explore_block(
                index=0,
                url=url,
                result=result,
                pattern_text=pattern or None,
                context_lines=context_lines,
                max_matches=max_matches_int,
                pattern=compiled_pattern,
                pattern_error=pattern_error,
            )
            return truncate_output(results_str, env._max_response_chars)

        async def open_lines(url: str, lines: list = None) -> str:
            """Get webpage content for a single URL.

            Args:
                url: URL to open.
                lines: Optional line ranges as [[start, end], ...] or a single [start, end].
                    Ranges are 0-based inclusive, sorted, and overlapping ranges are merged.
            """
            line_ranges = normalize_line_ranges(lines) if lines is not None else []
            use_line_ranges = lines is not None
            result = await open_one_result(url)
            is_error = (
                result.get("type") == "error" or result.get("format") == "error"
            )
            content = result.get("content")
            content_text = "" if content is None else str(content)

            if is_error:
                error_text = content_text or "error"
                if use_line_ranges:
                    range_lines = [
                        f"L{start}..{end}: (no content)"
                        for start, end in line_ranges
                    ]
                    out = (
                        "\n".join([error_text, *range_lines])
                        if range_lines
                        else error_text
                    )
                else:
                    out = error_text
            elif use_line_ranges:
                if not line_ranges:
                    out = "(no content)"
                elif not content_text:
                    out = "\n".join(
                        [
                            f"L{start}..{end}: (no content)"
                            for start, end in line_ranges
                        ]
                    )
                else:
                    out = render_line_ranges(content_text, line_ranges)
            else:
                out = content_text if content_text else "(no content)"

            return truncate_output(out, env._max_response_chars)

        tools: list = [search_web, scan_page, open_lines]

        if env._finish_with_tool:

            async def finish(final_answer: str) -> str:
                """Provide the final answer to the task. Stops execution."""
                if env._current_state is not None:
                    env._current_state.custom["done"] = True
                    env._current_state.custom["final_answer"] = final_answer
                return final_answer

            tools.append(finish)

        return tools

    # =========================================================================
    # Dataset
    # =========================================================================

    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        """Load DeepDive QA dataset from HuggingFace.

        Pass ``eval=True`` in kwargs to load the held-out evaluation split.
        """
        use_eval = kwargs.get("eval", False)
        raw_split = hf_load_dataset(self._dataset_name, split=self._dataset_split)

        split = raw_split.train_test_split(
            test_size=self._dataset_test_size, seed=self._dataset_seed
        )
        data = split["test"] if use_eval else split["train"]

        if num_samples > 0:
            data = data.select(range(min(num_samples, len(data))))

        samples: list[Sample] = []
        for item in data:
            question = (item.get("question") or "").rstrip()
            answer = (item.get("answer") or "").rstrip()

            prompt_text = question
            if not self._finish_with_tool:
                prompt_text += PROMPT_SUFFIX

            metadata: dict[str, Any] = {"raw_question": question}
            for key in METADATA_KEYS:
                if key in item and item[key] is not None:
                    metadata[key] = item[key]

            samples.append(
                Sample(
                    prompt=prompt_text,
                    answer=answer,
                    metadata=metadata,
                )
            )

        self._samples = samples
        return samples

    # =========================================================================
    # Rollout lifecycle
    # =========================================================================

    async def env_response(
        self, messages: list[ChatMessage], state: RolloutState
    ) -> list[ChatMessage]:
        """Execute tools with access to rollout state (needed by finish tool)."""
        self._current_state = state
        try:
            return await super().env_response(messages, state)
        finally:
            self._current_state = None

    def is_done(self, state: RolloutState) -> tuple[bool, str | None]:
        """Check if the finish tool was called or max turns reached."""
        if state.custom.get("done"):
            return True, "finish_called"
        return super().is_done(state)

    def is_final_answer(self, completion: str, state: RolloutState) -> bool:
        """Detect final answer via \\boxed{} (when not using finish tool) or no tool calls."""
        if not self._finish_with_tool:
            boxed = _extract_boxed_answer(completion)
            if boxed:
                state.custom["final_answer"] = boxed
                return True
        return super().is_final_answer(completion, state)

    # =========================================================================
    # Reward
    # =========================================================================

    async def compute_reward(
        self, state: RolloutState, eos_token: str = ""
    ) -> RewardResult:
        """Judge-based reward with optional redundancy penalty."""
        gold_answer = state.sample.answer

        # Early return for infrastructure errors (Serper API failures)
        for tr in state.custom.get("tool_results", []):
            if not tr.get("success") and "Serper API error" in (
                tr.get("result") or ""
            ):
                return RewardResult(
                    total_reward=0.0,
                    sample_metrics={"judge_reward": 0.0, "redundancy_penalty_reward": 0.0},
                    golden_answers={"judge_reward": gold_answer},
                    info_turns=[
                        {
                            "turn_order": 0,
                            "info_key": "serper_error",
                            "info_value": "Serper API error — reward set to 0",
                            "info_type": "text",
                        }
                    ],
                )

        # Build extra data
        final_answer = state.custom.get("final_answer") or self._get_last_completion(state)
        sample_tags: dict[str, str] = {}
        for tag_key in ("category", "difficulty", "source"):
            val = state.sample.metadata.get(tag_key)
            if val is not None:
                sample_tags[tag_key] = str(val)

        return await self.rubric.score(
            state=state,
            extra_info_turns=[
                {
                    "turn_order": 0,
                    "info_key": "final_answer",
                    "info_value": final_answer or "(none)",
                    "info_type": "text",
                }
            ],
            extra_sample_tags=sample_tags,
        )

    # -- rubric reward functions -------------------------------------------

    async def _judge_reward(self, state: RolloutState) -> tuple[float, str]:
        final_answer = state.custom.get("final_answer") or self._get_last_completion(state)
        gold = state.sample.answer
        raw_question = state.sample.metadata.get("raw_question", "")
        score = await self._judge_evaluate(raw_question, final_answer or "", gold)
        return score, gold

    def _redundancy_penalty_reward(self, state: RolloutState) -> float:
        return self._compute_redundancy(state)

    def _search_web_mean_queries(self, state: RolloutState) -> float:
        search_calls = [tc for tc in state.custom.get("tool_calls", []) if tc["name"] == "search_web"]
        total_queries = 0
        for tc in search_calls:
            queries = tc.get("arguments", {}).get("queries", [])
            if isinstance(queries, list):
                total_queries += min(len(queries), 10)
        return total_queries / len(search_calls) if search_calls else 0.0

    @staticmethod
    def _tool_error_rate(state: RolloutState, tool_name: str) -> float:
        tool_calls = state.custom.get("tool_calls", [])
        tool_results = state.custom.get("tool_results", [])
        calls = sum(1 for tc in tool_calls if tc["name"] == tool_name)
        errors = sum(1 for tr in tool_results if tr["name"] == tool_name and not tr["success"])
        return errors / calls if calls > 0 else 0.0

    def _search_web_error_rate(self, state: RolloutState) -> float:
        return self._tool_error_rate(state, "search_web")

    def _scan_page_error_rate(self, state: RolloutState) -> float:
        return self._tool_error_rate(state, "scan_page")

    def _open_lines_error_rate(self, state: RolloutState) -> float:
        return self._tool_error_rate(state, "open_lines")

    def _finish_error_rate(self, state: RolloutState) -> float:
        return self._tool_error_rate(state, "finish")

    # =========================================================================
    # Private helpers
    # =========================================================================

    async def _judge_evaluate(
        self, question: str, completion: str, answer: str
    ) -> float:
        """Call the judge model and return 1.0 (correct) or 0.0."""
        if not completion:
            return 0.0
        try:
            judge_response = await self._call_judge(question, completion, answer)
            return 1.0 if "yes" in judge_response.lower() else 0.0
        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}")
            return 0.0

    def _compute_redundancy(self, state: RolloutState) -> float:
        """Jaccard-similarity penalty across search_web queries."""
        tool_calls = state.custom.get("tool_calls", [])
        token_re = re.compile(r"\w+")
        query_sets: list[set[str]] = []

        for tc in tool_calls:
            if tc["name"] != "search_web":
                continue
            args = tc.get("arguments", {})
            queries = args.get("queries", [])
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list):
                continue
            for q in queries:
                if isinstance(q, str) and q.strip():
                    tokens = {t.lower() for t in token_re.findall(q.strip())}
                    if tokens:
                        query_sets.append(tokens)

        if len(query_sets) < 2:
            return 0.0

        sim_sum = 0.0
        pair_count = 0
        for i in range(len(query_sets)):
            for j in range(i + 1, len(query_sets)):
                s1, s2 = query_sets[i], query_sets[j]
                sim_sum += len(s1 & s2) / len(s1 | s2)
                pair_count += 1

        return sim_sum / pair_count if pair_count > 0 else 0.0

    def _get_last_completion(self, state: RolloutState) -> str | None:
        """Extract the last assistant completion text from the trajectory."""
        if not state.trajectory:
            return None
        last_step = state.trajectory[-1]
        completion = last_step.completion
        if isinstance(completion, list):
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
        elif isinstance(completion, str):
            return completion
        return None

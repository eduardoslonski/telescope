"""Rollout logic for inference servers."""
import asyncio
import http.client as _httpclib
import json as _json_mod
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from telescope.utils import config
from telescope.environments.base import (
    Sample, 
    RewardResult, 
    MultiTurnEnvironment,
    RolloutState,
    TrajectoryStep,
    ChatMessage,
)
from telescope.utils.tlog import get_logger

_log = get_logger("generate")

# ---------------------------------------------------------------------------
# Thread-pool HTTP: moves socket I/O off the asyncio event loop.
#
# With 2000+ concurrent requests, async httpx creates massive event loop
# contention (~3s overhead per request from JSON parsing + protocol handling).
# Running HTTP in threads with stdlib http.client avoids this:
#   - http.client is a thin socket wrapper; socket I/O releases the GIL.
#   - No connection pool locks (unlike httpx), so 2000+ threads don't fight.
#   - JSON parsing stays on the event loop (single-threaded, no GIL contention).
#   - Pool size must be >= total concurrent requests (threads block for the
#     full I/O duration of each request).
# ---------------------------------------------------------------------------
_http_pool = ThreadPoolExecutor(max_workers=2048, thread_name_prefix="http")


def _sync_http_post(url: str, body: bytes) -> tuple[bytes, dict[str, str], float]:
    """Synchronous HTTP POST using stdlib http.client — runs in the thread pool."""
    parsed = urlparse(url)
    _t0 = time.time()
    conn = _httpclib.HTTPConnection(parsed.hostname, parsed.port, timeout=1200)
    try:
        conn.request("POST", parsed.path, body=body,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(body))})
        resp = conn.getresponse()
        resp_body = resp.read()
        headers = {k.lower(): v for k, v in resp.getheaders()}
    finally:
        conn.close()
    _dur = time.time() - _t0
    return resp_body, headers, _dur


def get_chat_template_kwargs() -> dict:
    """
    Build chat template kwargs from config.
    
    This centralizes the thinking mode configuration so all apply_chat_template
    calls use consistent settings.
    
    Returns:
        Dict of kwargs to pass to tokenizer.apply_chat_template()
    """
    kwargs = {}
    
    # Thinking mode (for Qwen3 and similar models)
    # When enable_thinking=True, model generates <think>...</think> blocks
    enable_thinking = config.cfg.enable_thinking
    kwargs['enable_thinking'] = enable_thinking
    
    if enable_thinking:
        _log.debug("Thinking mode enabled in chat template")
    
    return kwargs

# Retry configuration for HTTP requests
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds

# Set to True during graceful shutdown to abort retries immediately
_shutting_down = False


@dataclass
class SampleLifecycleCallbacks:
    """Callbacks for sample lifecycle events, created per-sample by the orchestrator.

    The orchestrator builds closures that capture request context
    (request_id, sample_idx, server info) and log the appropriate events.
    """
    on_generation_start: Callable[[int], None] | None = None  # Log generation start (receives generation_idx)
    on_generation_end: Callable[[int], None] | None = None    # Log generation end (receives generation_idx)
    on_inference_end: Callable[[], None] | None = None    # Free lane slot (no longer logs generation end)
    on_reward_start: Callable[[], None] | None = None     # Log compute_reward_start orchestrator event
    on_reward_end: Callable[[float], None] | None = None   # Log compute_reward_end (receives duration in seconds)
    on_env_response_start: Callable[[], None] | None = None  # Log env_response_start orchestrator event
    on_env_response_end: Callable[[], None] | None = None    # Log env_response_end orchestrator event


# =============================================================================
# Local Tokenization Helper
# =============================================================================

@dataclass
class InterleavedTokenizer:
    """
    Handles local tokenization for interleaved multi-turn rollouts.

    The key insight is that we can compute suffix_ids and rollout_prompt_ids
    once at initialization, then use them to build token sequences locally.
    """
    tokenizer: Any
    suffix_ids: list[int]  # Tokens added by chat template after assistant messages
    rollout_prompt_ids: list[int]  # Tokens to add before next assistant rollout
    base_conversation_ids: list[int]  # Fixed base for slicing env responses
    eos_token_id: int
    chat_template_kwargs: dict  # Stored kwargs for consistent apply_chat_template calls
    
    @classmethod
    def from_tokenizer(cls, tokenizer: Any, chat_template_kwargs: dict = None) -> "InterleavedTokenizer":
        """
        Initialize from a HuggingFace tokenizer.
        
        Computes suffix_ids using a fixed-base approach:
        1. Create a dummy conversation
        2. Tokenize it
        3. Find what tokens appear after the assistant content
        
        Args:
            tokenizer: HuggingFace tokenizer with chat_template support
            chat_template_kwargs: Optional kwargs for apply_chat_template
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for interleaved rollouts")
        
        chat_template_kwargs = chat_template_kwargs or {}
        eos_token_id = tokenizer.eos_token_id
        
        # === Compute suffix_ids (tokens after assistant message) ===
        dummy_content = "World!"
        
        # Tokenize just the raw content
        content_ids = tokenizer.encode(dummy_content, add_special_tokens=False)
        
        # Tokenize full conversation with template (no rollout prompt)
        dummy_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": dummy_content},
        ]
        full_ids = tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=True,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )

        # Find suffix: tokens after the last occurrence of content's last token
        suffix_ids = []
        if content_ids:
            last_content_token = content_ids[-1]
            for i in range(len(full_ids) - 1, -1, -1):
                if full_ids[i] == last_content_token:
                    suffix_ids = full_ids[i + 1:]
                    break

        # === Compute rollout_prompt_ids ===
        # These are the tokens that start the assistant turn (e.g., "<|assistant|>\n")
        messages_with_gen = tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        messages_without_gen = tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=True,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        rollout_prompt_ids = messages_with_gen[len(messages_without_gen):]

        # === Compute base_conversation_ids for fixed-base slicing ===
        # Used to tokenize env responses by: tokenize([base + env]) then slice off base
        base_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
        ]
        base_ids = tokenizer.apply_chat_template(
            base_conversation,
            tokenize=True,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        # Trim to last EOS so env response can capture its own EOS
        if eos_token_id in base_ids:
            last_eos_idx = len(base_ids) - 1 - base_ids[::-1].index(eos_token_id)
            base_ids = base_ids[:last_eos_idx + 1]
        
        _log.info(f"InterleavedTokenizer initialized: suffix_ids={suffix_ids}, gen_prompt_ids={rollout_prompt_ids}")
        
        return cls(
            tokenizer=tokenizer,
            suffix_ids=suffix_ids,
            rollout_prompt_ids=rollout_prompt_ids,
            base_conversation_ids=base_ids,
            eos_token_id=eos_token_id,
            chat_template_kwargs=chat_template_kwargs,
        )
    
    def get_env_response_ids(
        self, 
        env_messages: list[ChatMessage], 
        add_rollout_prompt: bool = True,
        chat_template_kwargs: dict = None,
    ) -> list[int]:
        """
        Tokenize environment response messages using the fixed-base approach.
        
        Instead of calling /tokenize, we:
        1. Tokenize [base_conversation + env_messages]
        2. Slice off the base_conversation_ids
        
        This gives us the exact tokens for the env response in chat template format.
        
        Args:
            env_messages: The environment's response messages
            add_rollout_prompt: Whether to add assistant prompt for next turn
            chat_template_kwargs: Optional kwargs for apply_chat_template (defaults to stored kwargs)
        """
        if not env_messages:
            return self.rollout_prompt_ids if add_rollout_prompt else []
        
        # Use stored kwargs if not explicitly provided (ensures enable_thinking is consistent)
        chat_template_kwargs = chat_template_kwargs if chat_template_kwargs is not None else self.chat_template_kwargs
        
        # Build: base_conversation + env_messages
        base_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
        ]
        full_messages = base_conversation + env_messages
        
        # Tokenize full conversation
        full_ids = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=add_rollout_prompt,
            **chat_template_kwargs,
        )
        
        # Slice off the base to get just env response + rollout prompt
        env_ids = full_ids[len(self.base_conversation_ids):]
        
        return env_ids
    
    def build_next_turn_prompt_ids(
        self,
        running_token_ids: list[int],
        prev_completion_ids: list[int],
        env_messages: list[ChatMessage],
        chat_template_kwargs: dict = None,
    ) -> list[int]:
        """
        Build the token IDs for the next turn's prompt.
        
        This implements the "exact prefix invariant" for importance sampling:
        prompt_n+1 = running_tokens + suffix (with overlap handled) + env_response_ids
        
        Args:
            running_token_ids: All tokens so far (prompt + completions + env responses)
            prev_completion_ids: The previous turn's completion token IDs
            env_messages: New environment response messages
            chat_template_kwargs: Optional kwargs for apply_chat_template (defaults to stored kwargs)
        """
        # Handle suffix overlap (model may have already generated some suffix tokens)
        overlap_len = _find_largest_overlap(prev_completion_ids, self.suffix_ids)
        suffix_to_add = self.suffix_ids[overlap_len:]
        
        # Get env response tokens
        env_ids = self.get_env_response_ids(
            env_messages, 
            add_rollout_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        
        # Concatenate: running + suffix + env_response
        return running_token_ids + suffix_to_add + env_ids


# Global interleaved tokenizer (initialized lazily)
_interleaved_tokenizer: InterleavedTokenizer | None = None


def init_interleaved_tokenizer(tokenizer: Any, chat_template_kwargs: dict = None) -> InterleavedTokenizer:
    """
    Initialize the global interleaved tokenizer.
    
    If chat_template_kwargs is not provided, uses config-based defaults
    (including ENABLE_THINKING setting).
    """
    global _interleaved_tokenizer
    
    # Use config-based defaults if not explicitly provided
    if chat_template_kwargs is None:
        chat_template_kwargs = get_chat_template_kwargs()
    
    _interleaved_tokenizer = InterleavedTokenizer.from_tokenizer(tokenizer, chat_template_kwargs)
    return _interleaved_tokenizer


def get_interleaved_tokenizer() -> InterleavedTokenizer | None:
    """Get the global interleaved tokenizer (may be None if not initialized)."""
    return _interleaved_tokenizer


# =============================================================================
# HTTP Request Utilities
# =============================================================================

async def _retry_request(coro_factory, max_retries=MAX_RETRIES):
    """
    Retry HTTP request with exponential backoff on connection errors.
    
    Args:
        coro_factory: A callable that returns a new coroutine for each attempt
        max_retries: Maximum number of retry attempts
        
    Returns:
        The result of the successful request
        
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        if _shutting_down:
            raise RuntimeError("Shutting down")
        try:
            return await coro_factory()
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                ConnectionError, OSError, _httpclib.HTTPException) as e:
            last_exception = e
            if _shutting_down:
                raise
            if attempt < max_retries:
                backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                _log.warning(f"HTTP request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff}s: {e}")
                await asyncio.sleep(backoff)
            else:
                _log.error(f"HTTP request failed after {max_retries + 1} attempts: {e}")
    raise last_exception


class PromptTooLongError(Exception):
    """Raised when a prompt exceeds the model's context length limit."""
    def __init__(self, prompt_tokens: int, max_tokens: int, message: str = None):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.message = message or f"Prompt has {prompt_tokens} tokens but max is {max_tokens}"
        super().__init__(self.message)


class RolloutError(Exception):
    """Raised when vLLM returns an error response."""
    def __init__(self, error_type: str, message: str, response_data: dict = None):
        self.error_type = error_type
        self.response_data = response_data
        super().__init__(f"{error_type}: {message}")


class ContextExhaustedError(Exception):
    """Raised when the context window is too full to generate meaningful output."""
    def __init__(self, prompt_tokens: int, max_context: int, requested_tokens: int, message: str = None):
        self.prompt_tokens = prompt_tokens
        self.max_context = max_context
        self.requested_tokens = requested_tokens
        self.available_tokens = max_context - prompt_tokens
        self.message = message or (
            f"Context exhausted: {prompt_tokens} prompt tokens + {requested_tokens} requested "
            f"> {max_context} max context (only {self.available_tokens} available)"
        )
        super().__init__(self.message)


def _parse_context_length_error(error_message: str) -> tuple[int, int] | None:
    """
    Parse vLLM context length error to extract token counts.
    
    Example error: "This model's maximum context length is 1248 tokens. 
                   However, your request has 2069 input tokens."
    
    Returns:
        Tuple of (max_context_tokens, request_tokens) or None if not a context length error
    """
    import re
    max_match = re.search(r"maximum context length is (\d+) tokens", error_message)
    request_match = re.search(r"request has (\d+) input tokens", error_message)
    
    if max_match and request_match:
        return int(max_match.group(1)), int(request_match.group(1))
    return None


def _parse_max_tokens_error(error_message: str) -> tuple[int, int, int] | None:
    """
    Parse vLLM error about max_tokens being too large for remaining context.
    
    Example error: "'max_tokens' or 'max_completion_tokens' is too large: 800. 
                   This model's maximum context length is 2048 tokens and your 
                   request has 1459 input tokens (800 > 2048 - 1459)."
    
    Returns:
        Tuple of (max_context, prompt_tokens, requested_tokens) or None
    """
    import re
    
    # Match: "maximum context length is X tokens"
    max_match = re.search(r"maximum context length is (\d+) tokens", error_message)
    # Match: "request has X input tokens"
    input_match = re.search(r"has (\d+) input tokens", error_message)
    # Match: "is too large: X"
    requested_match = re.search(r"is too large: (\d+)", error_message)
    
    if max_match and input_match and requested_match:
        return (
            int(max_match.group(1)),
            int(input_match.group(1)),
            int(requested_match.group(1))
        )
    return None


async def generate_completion_with_tokens(
    client: httpx.AsyncClient,
    prompt_token_ids: list[int],
    server_url: str,
    max_tokens: int | None = None,
    priority: int = 0,
    sampling_params: dict | None = None,
    timing_out: dict | None = None,
) -> tuple[dict, float, float, int]:
    """
    Generate a completion from pre-tokenized prompt.

    This is used for interleaved rollouts where we want to avoid re-tokenizing
    the conversation history to maintain exact token prefix matching.

    Dynamically reduces max_tokens if prompt + max_tokens > MAX_MODEL_LEN.

    Args:
        client: HTTP client
        prompt_token_ids: Pre-tokenized prompt as list of token IDs
        server_url: vLLM server URL
        max_tokens: Max tokens to generate
        sampling_params: Optional override for config.cfg.get_sampling_params()

    Returns:
        Tuple of (response_data, start_time, end_time, actual_max_tokens)
    """
    # Calculate available tokens and adjust max_tokens dynamically
    prompt_len = len(prompt_token_ids)
    available_tokens = config.cfg.max_model_len - prompt_len

    if available_tokens <= 0:
        # Prompt itself exceeds context - can't generate anything
        raise PromptTooLongError(prompt_len, config.cfg.max_model_len,
            f"Prompt has {prompt_len} tokens but max context is {config.cfg.max_model_len}")

    # Use min of requested and available
    sampling = sampling_params if sampling_params is not None else config.cfg.get_sampling_params()
    requested_tokens = max_tokens if max_tokens is not None else sampling.get("max_tokens")
    if requested_tokens is None:
        raise ValueError(
            "max_tokens is required but was not set. "
            "Pass max_tokens explicitly or set it in sampling_params / config."
        )
    actual_max_tokens = min(requested_tokens, available_tokens)

    if actual_max_tokens < requested_tokens:
        _log.debug(f"Reduced max_tokens from {requested_tokens} to {actual_max_tokens} "
                   f"(prompt={prompt_len}, available={available_tokens})")

    data = {
        "model": config.cfg.model,
        "prompt": [prompt_token_ids],  # vLLM expects list of lists for token prompts
        "return_token_ids": True,
        "n": 1,
        "logprobs": 1,
        "skip_special_tokens": False,  # Keep special tokens in text to match token_ids
        "include_stop_str_in_output": False,  # Ensure EOS/stop tokens appear in text
        **sampling,
        "max_tokens": actual_max_tokens,  # always override with the dynamically adjusted value
        "priority": priority,
    }

    start_time = time.time()
    if timing_out is not None:
        if "start_time" not in timing_out:
            timing_out["start_time"] = start_time

    _req_body = _json_mod.dumps(data).encode()
    _url = f"{server_url}/v1/completions"
    loop = asyncio.get_event_loop()

    async def make_request():
        _resp_body, _, _ = await loop.run_in_executor(
            _http_pool, _sync_http_post, _url, _req_body,
        )
        return _json_mod.loads(_resp_body)

    try:
        result = await _retry_request(make_request)
    except asyncio.CancelledError:
        if timing_out is not None:
            timing_out["end_time"] = time.time()
        raise
    end_time = time.time()
    if timing_out is not None:
        timing_out["end_time"] = end_time

    # Check for error response
    if "choices" not in result:
        error_msg = result.get("message", result.get("detail", str(result)))
        error_type = result.get("type", "unknown_error")

        max_tokens_info = _parse_max_tokens_error(error_msg)
        if max_tokens_info:
            max_ctx, prompt_tokens, requested = max_tokens_info
            raise ContextExhaustedError(prompt_tokens, max_ctx, requested, error_msg)

        context_info = _parse_context_length_error(error_msg)
        if context_info:
            max_ctx, request_tokens = context_info
            raise PromptTooLongError(request_tokens, max_ctx, error_msg)

        raise RolloutError(error_type, error_msg, result)

    return result, start_time, end_time, actual_max_tokens


async def generate_completion(
    client: httpx.AsyncClient,
    prompt: str,
    server_url: str,
    max_tokens: int | None = None,
    prompt_token_count: int | None = None,
    priority: int = 0,
    sampling_params: dict | None = None,
    timing_out: dict | None = None,
) -> tuple[dict, float, float, int]:
    """
    Generate a completion from a specific inference server.

    If prompt_token_count is provided, dynamically reduces max_tokens to fit
    within MAX_MODEL_LEN.

    Args:
        client: HTTP client
        prompt: The prompt string
        server_url: vLLM server URL
        max_tokens: Max tokens to generate (defaults to SAMPLING_PARAMS["max_tokens"])
        prompt_token_count: Optional prompt token count for dynamic max_tokens adjustment
        sampling_params: Optional override for config.cfg.get_sampling_params()

    Returns:
        Tuple of (response_data, start_time, end_time, actual_max_tokens)

    Raises:
        PromptTooLongError: If the prompt exceeds the model's context length
        ContextExhaustedError: If max_tokens requested exceeds remaining context
        RolloutError: If vLLM returns another type of error
    """
    sampling = sampling_params if sampling_params is not None else config.cfg.get_sampling_params()
    requested_tokens = max_tokens if max_tokens is not None else sampling.get("max_tokens")
    if requested_tokens is None:
        raise ValueError(
            "max_tokens is required but was not set. "
            "Pass max_tokens explicitly or set it in sampling_params / config."
        )
    actual_max_tokens = requested_tokens

    # If we know the prompt token count, dynamically adjust max_tokens
    if prompt_token_count is not None:
        available_tokens = config.cfg.max_model_len - prompt_token_count

        if available_tokens <= 0:
            raise PromptTooLongError(prompt_token_count, config.cfg.max_model_len,
                f"Prompt has {prompt_token_count} tokens but max context is {config.cfg.max_model_len}")
        
        actual_max_tokens = min(requested_tokens, available_tokens)
        
        if actual_max_tokens < requested_tokens:
            _log.debug(f"Reduced max_tokens from {requested_tokens} to {actual_max_tokens} "
                       f"(prompt={prompt_token_count}, available={available_tokens})")
    
    data = {
        "model": config.cfg.model,
        "prompt": [prompt],
        "return_token_ids": True,
        "n": 1,
        "logprobs": 1,  # Request logprobs for TIS/PPO clipping
        "skip_special_tokens": False,  # Keep special tokens in text to match token_ids
        "include_stop_str_in_output": False,  # Ensure EOS/stop tokens appear in text
        **sampling,
        "max_tokens": actual_max_tokens,  # always override with the dynamically adjusted value
        "priority": priority,
    }

    start_time = time.time()
    if timing_out is not None:
        if "start_time" not in timing_out:
            timing_out["start_time"] = start_time

    _req_body = _json_mod.dumps(data).encode()
    _url = f"{server_url}/v1/completions"
    loop = asyncio.get_event_loop()

    async def make_request():
        _resp_body, _, _ = await loop.run_in_executor(
            _http_pool, _sync_http_post, _url, _req_body,
        )
        return _json_mod.loads(_resp_body)

    try:
        result = await _retry_request(make_request)
    except asyncio.CancelledError:
        if timing_out is not None:
            timing_out["end_time"] = time.time()
        raise
    end_time = time.time()
    if timing_out is not None:
        timing_out["end_time"] = end_time

    # Check for error response (vLLM returns error info instead of choices)
    if "choices" not in result:
        error_msg = result.get("message", result.get("detail", str(result)))
        error_type = result.get("type", "unknown_error")

        # Check if this is a "max_tokens too large" error (context exhausted)
        max_tokens_info = _parse_max_tokens_error(error_msg)
        if max_tokens_info:
            max_ctx, prompt_tokens, requested = max_tokens_info
            raise ContextExhaustedError(prompt_tokens, max_ctx, requested, error_msg)

        # Check if this is a context length error (prompt itself too long)
        context_info = _parse_context_length_error(error_msg)
        if context_info:
            max_ctx, request_tokens = context_info
            raise PromptTooLongError(request_tokens, max_ctx, error_msg)

        raise RolloutError(error_type, error_msg, result)

    return result, start_time, end_time, actual_max_tokens


def _format_messages_to_prompt(messages: list[ChatMessage], tokenizer: Any) -> str:
    """
    Format chat messages into a prompt string using the tokenizer's chat template.
    
    This allows us to use the completions API (which returns token IDs) while
    still supporting multi-turn conversations.
    
    Uses config-based chat template kwargs (e.g., enable_thinking for Qwen3).
    
    Args:
        messages: List of chat messages
        tokenizer: HuggingFace tokenizer with chat template
        
    Returns:
        Formatted prompt string
    """
    if tokenizer is None:
        # Fallback: simple formatting without chat template
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|{role}|>\n{content}\n"
        formatted += "<|assistant|>\n"
        return formatted
    
    # Use tokenizer's chat template with config-based kwargs
    chat_kwargs = get_chat_template_kwargs()
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Add the assistant prompt prefix
        **chat_kwargs,
    )


async def run_multiturn_rollout(
    client: httpx.AsyncClient,
    env: MultiTurnEnvironment,
    sample: Sample,
    server_url: str,
    tokenizer: Any = None,
    prefetched_messages: list[ChatMessage] | None = None,
    prefetched_prompt_str: str | None = None,
    prefetched_prompt_token_count: int | None = None,
    sampling_params: dict | None = None,
    interleaved: bool | None = None,
    timing_out: dict | None = None,
    lifecycle: SampleLifecycleCallbacks | None = None,
) -> tuple[RolloutState, list[dict]]:
    """
    Run a complete multi-turn rollout with an environment.

    When INTERLEAVED_ROLLOUTS is enabled (recommended), uses exact token reuse
    with LOCAL tokenization (no HTTP calls to /tokenize):
    - First turn: format messages → generate → save prompt_ids + completion_ids
    - Subsequent turns: reuse previous tokens + local tokenize env response → generate

    This ensures the "exact prefix invariant" for proper importance sampling in RL.

    When INTERLEAVED_ROLLOUTS is disabled, re-tokenizes the full conversation each turn
    (simpler but may cause token mismatch issues with some chat templates).

    Args:
        client: HTTP client for rollout requests
        env: Multi-turn environment instance
        sample: Input sample with prompt
        server_url: Inference server URL
        tokenizer: HuggingFace tokenizer for formatting chat template
        prefetched_messages: Optional precomputed initial messages
        prefetched_prompt_str: Optional precomputed prompt string for first turn
        prefetched_prompt_token_count: Optional token count for precomputed prompt
        sampling_params: Optional override for config.cfg.get_sampling_params() (used by evals)
        interleaved: Optional override for config.cfg.interleaved_rollouts (used by evals)

    Returns:
        Tuple of (final_state, request_timings)
    """
    # Initialize state
    state = env.create_initial_state(sample)
    request_timings = []

    # Get initial prompt messages
    messages = prefetched_messages if prefetched_messages is not None else env.get_initial_prompt(sample, tokenizer)

    # For interleaved rollouts: track the running token sequence
    # Structure: prompt₀ + completion₀ + [suffix + env_response + completion]₁ + ...
    running_token_ids: list[int] = []

    # Track turns for logging (model rollouts + env responses with order and type)
    logged_turns: list[dict] = []
    turn_order = 0

    # Generation-centric data collection
    generation_idx = 0
    logged_generations: list[dict] = []
    logged_env_responses: list[dict] = []
    logged_tool_calls: list[dict] = []

    use_interleaved = interleaved if interleaved is not None else config.cfg.interleaved_rollouts
    
    # Get the interleaved tokenizer for local tokenization (no HTTP calls!)
    interleaved_tok = get_interleaved_tokenizer() if use_interleaved else None
    if use_interleaved and interleaved_tok is None:
        _log.warning("INTERLEAVED_ROLLOUTS enabled but tokenizer not initialized, falling back to non-interleaved")
        use_interleaved = False
    
    try:
        while True:
            # Log per-turn generation start
            if lifecycle is not None and lifecycle.on_generation_start is not None:
                lifecycle.on_generation_start(generation_idx)

            if state.num_turns == 0 or not use_interleaved:
                # First turn OR non-interleaved: use string-based rollout
                if state.num_turns == 0 and prefetched_prompt_str is not None:
                    prompt_str = prefetched_prompt_str
                    prompt_token_count = prefetched_prompt_token_count
                else:
                    prompt_str = _format_messages_to_prompt(messages, tokenizer)
                    # Get prompt token count for dynamic max_tokens adjustment
                    prompt_token_count = len(tokenizer.encode(prompt_str)) if tokenizer else None
                result, start_time, end_time, req_max_tokens = await generate_completion(
                    client, prompt_str, server_url, prompt_token_count=prompt_token_count,
                    priority=-state.num_turns,
                    sampling_params=sampling_params,
                    timing_out=timing_out,
                )
            else:
                # Subsequent turns with interleaved: use token-based rollout
                # All tokenization is done LOCALLY using InterleavedTokenizer

                # Get env response messages (everything after the previous turn)
                prev_turn_messages = state.trajectory[-1].prompt
                if isinstance(prev_turn_messages, list):
                    prev_len = len(prev_turn_messages) + 1  # +1 for assistant response
                else:
                    prev_len = 1
                env_response_messages = messages[prev_len:]

                # Get previous turn's completion for overlap detection
                prev_turn_completion_ids = state.trajectory[-1].completion_token_ids

                # Build prompt token IDs using LOCAL tokenization
                prompt_token_ids_for_turn = interleaved_tok.build_next_turn_prompt_ids(
                    running_token_ids=running_token_ids,
                    prev_completion_ids=prev_turn_completion_ids,
                    env_messages=env_response_messages,
                )

                # Generate using pre-tokenized prompt
                result, start_time, end_time, req_max_tokens = await generate_completion_with_tokens(
                    client, prompt_token_ids_for_turn, server_url,
                    priority=-state.num_turns,
                    sampling_params=sampling_params,
                    timing_out=timing_out,
                )
            
            # Extract completion data
            choice = result["choices"][0]
            completion_text = choice.get("text", "")
            completion_token_ids = choice.get("token_ids", [])
            prompt_token_ids = choice.get("prompt_token_ids", [])
            finish_reason = choice.get("finish_reason", "")
            is_truncated = finish_reason == "length"
            
            # Extract logprobs
            logprobs_data = choice.get("logprobs", {})
            token_logprobs = logprobs_data.get("token_logprobs", []) if logprobs_data else []
            completion_logprobs = [lp if lp is not None else 0.0 for lp in token_logprobs]
            
            # Track running token sequence for full conversation reconstruction
            running_token_ids = prompt_token_ids + completion_token_ids
            
            # Get completion tokens for this turn from usage
            usage = result.get("usage", {})
            turn_completion_tokens = usage.get("completion_tokens", 0)
            
            # Log model rollout turn with token count and stop reason
            logged_turns.append({
                "turn_order": turn_order,
                "turn_type": "model",
                "content": completion_text,
                "tokens": turn_completion_tokens,
                "stop_reason": finish_reason,
            })
            turn_order += 1

            # Generation-centric logging
            logged_generations.append({
                "generation_idx": generation_idx,
                "content": completion_text,
                "tokens": turn_completion_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "tool_call_count": 0,  # Updated below after env_response parses tool calls
                "stop_reason": finish_reason,
            })
            
            # Build trajectory step
            step = TrajectoryStep(
                prompt=messages.copy(),
                completion=[{"role": "assistant", "content": completion_text}],
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                completion_logprobs=completion_logprobs,
                is_truncated=is_truncated,
            )
            state.trajectory.append(step)
            
            # Track request timing (including vLLM request ID for tracing correlation)
            vllm_request_id = result.get("id", "")
            request_timings.append({
                "start_time": start_time,
                "end_time": end_time,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "rollout_tokens": usage.get("completion_tokens", 0),
                "turn": state.num_turns,
                "vllm_request_id": vllm_request_id,
                "max_tokens": req_max_tokens,
            })
            
            # Log per-turn generation end (LLM call complete, before env_response)
            if lifecycle is not None and lifecycle.on_generation_end is not None:
                lifecycle.on_generation_end(generation_idx)

            # Get environment response for next turn
            # env_response processes the model's action (e.g. evaluates a guess,
            # executes tool calls) and returns feedback messages.
            full_messages = messages + [{"role": "assistant", "content": completion_text}]
            if lifecycle is not None and lifecycle.on_env_response_start is not None:
                lifecycle.on_env_response_start()
            _env_resp_t0 = time.monotonic()
            env_messages = await env.env_response(full_messages, state)
            _env_resp_time = time.monotonic() - _env_resp_t0
            if lifecycle is not None and lifecycle.on_env_response_end is not None:
                lifecycle.on_env_response_end()

            # Termination is controlled exclusively by is_done().
            is_done, stop_reason = env.is_done(state)

            if is_done:
                state.is_completed = True
                state.stop_reason = stop_reason
                break

            # Log env response turn(s) for mid-rollout feedback
            if env_messages:
                env_turn_type = "env_response"
                if isinstance(env_messages[0], dict):
                    env_turn_type = env_messages[0].get("turn_type", "env_response")

                env_content = "\n".join(
                    msg.get("content", "") for msg in env_messages
                    if isinstance(msg, dict) and msg.get("content")
                )
                if env_content:
                    env_tokens = 0
                    if tokenizer is not None:
                        env_tokens = len(tokenizer.encode(env_content))

                    logged_turns.append({
                        "turn_order": turn_order,
                        "turn_type": env_turn_type,
                        "content": env_content,
                        "tokens": env_tokens,
                        "environment_response_time": _env_resp_time,
                    })
                    turn_order += 1

                    # Generation-centric env response logging
                    logged_env_responses.append({
                        "generation_idx": generation_idx,
                        "content": env_content,
                        "turn_type": env_turn_type,
                        "tokens": env_tokens,
                        "response_time": _env_resp_time,
                    })

            # Extract tool calls from state.custom if the environment tracks them
            # (ToolEnvironment populates these during env_response)
            new_tool_calls = state.custom.get("_pending_tool_calls", [])
            if new_tool_calls:
                # Update the generation's tool_call_count
                if logged_generations:
                    logged_generations[-1]["tool_call_count"] = len(new_tool_calls)
                for tc_idx, tc in enumerate(new_tool_calls):
                    logged_tool_calls.append({
                        "generation_idx": generation_idx,
                        "tool_call_idx": tc_idx,
                        "env_response_generation_idx": generation_idx,
                        "tool_name": tc.get("tool_name", tc.get("name", "")),
                        "arguments": tc.get("arguments", ""),
                        "raw_text": tc.get("raw_text", tc.get("raw", "")),
                        "result": tc.get("result", ""),
                        "success": tc.get("success", True),
                        "error": tc.get("error", ""),
                        "exit_code": tc.get("exit_code", -1),
                        "truncated": tc.get("truncated", False),
                        "result_tokens": tc.get("result_tokens", 0),
                        "sandbox_id": tc.get("sandbox_id", ""),
                    })
                # Clear pending tool calls for next generation
                state.custom["_pending_tool_calls"] = []

            generation_idx += 1

            # Build next prompt messages (for message tracking and non-interleaved mode)
            messages = env.get_next_prompt_messages(state, env_messages)
            
    except ContextExhaustedError as e:
        state.is_completed = True
        state.stop_reason = "context_exhausted"
    except PromptTooLongError as e:
        # Don't set state.error - keep partial trajectory (like ContextExhaustedError)
        # This happens when env response pushes the next prompt over the limit
        # Single-turn handles this separately in process_sample() and discards the group
        state.is_completed = True
        state.stop_reason = "prompt_too_long"
    except RolloutError as e:
        state.error = f"rollout_error: {str(e)}"
        state.is_completed = True
        state.stop_reason = "rollout_error"
    
    # Store data in state for downstream consumption
    state.custom["_logged_turns"] = logged_turns
    state.custom["_logged_generations"] = logged_generations
    state.custom["_logged_env_responses"] = logged_env_responses
    state.custom["_logged_tool_calls"] = logged_tool_calls
    # Store full token sequence (prompt + completions + env responses) for raw_string
    state.custom["_full_token_ids"] = running_token_ids
    
    return state, request_timings


def _find_largest_overlap(a: list[int], b: list[int]) -> int:
    """
    Find the largest overlapping sequence between the end of a and beginning of b.
    
    This is used to handle chat template suffix tokens that may already be
    present at the end of the completion.
    """
    if not a or not b:
        return 0
    
    max_possible = min(len(a), len(b))
    for overlap_len in reversed(range(1, max_possible + 1)):
        a_suffix = a[-overlap_len:]
        b_prefix = b[:overlap_len]
        
        if a_suffix == b_prefix:
            return overlap_len
    
    return 0


def _interleave_trajectory(
    state: RolloutState,
) -> tuple[list[int], list[int], list[int], list[float]]:
    """
    Build interleaved completion data from a multi-turn trajectory.

    Env response tokens are inserted between model completions so that the trainer sees the full
    conversation.  A ``completion_mask`` marks which tokens the model actually
    generated (1) vs environment response tokens that should be masked out
    during training (0).

    Returns:
        ``(prompt_ids, completion_ids, completion_mask, logprobs)`` where
        ``completion_ids`` is the interleaved sequence, ``completion_mask``
        has the same length with 1/0 for model/env tokens, and ``logprobs``
        contains the model logprobs for model tokens and 0.0 for env tokens.
    """
    trajectory = state.trajectory
    if not trajectory:
        return [], [], [], []

    # First step: use prompt + completion directly
    first = trajectory[0]
    prompt_ids = list(first.prompt_token_ids)
    completion_ids = list(first.completion_token_ids)
    completion_mask = [1] * len(completion_ids)
    logprobs = list(first.completion_logprobs)

    # Build prefix (everything the model has "seen" up to this point)
    prefix = list(first.prompt_token_ids) + list(first.completion_token_ids)

    for step in trajectory[1:]:
        # The step's prompt_token_ids contain the full conversation up to this
        # turn, including the env response.  The env response tokens are the
        # NEW tokens beyond the previous prefix.
        step_prompt_ids = list(step.prompt_token_ids)

        # Verify the prefix invariant: step's prompt should start with the
        # previous prefix.  A mismatch means the tokenizer produced different
        # tokens when re-encoding the conversation (can happen in non-interleaved
        # mode).  Log a warning so the issue is visible but continue anyway.
        if step_prompt_ids[:len(prefix)] != prefix:
            _log.warning(
                "Prefix mismatch in _interleave_trajectory: "
                f"expected {len(prefix)} prefix tokens to match, "
                f"but step prompt ({len(step_prompt_ids)} tokens) diverges. "
                "This can cause incorrect completion_mask values. "
                "Enable interleaved_rollouts to avoid this issue."
            )

        env_response_ids = step_prompt_ids[len(prefix):]

        # Extend completion with env response tokens (masked out)
        completion_ids.extend(env_response_ids)
        completion_mask.extend([0] * len(env_response_ids))
        logprobs.extend([0.0] * len(env_response_ids))

        # Extend completion with model completion tokens (trained on)
        step_completion_ids = list(step.completion_token_ids)
        step_logprobs = list(step.completion_logprobs)
        completion_ids.extend(step_completion_ids)
        completion_mask.extend([1] * len(step_completion_ids))
        logprobs.extend(step_logprobs)

        # Update prefix for next iteration
        prefix = step_prompt_ids + step_completion_ids

    assert len(completion_mask) == len(completion_ids), (
        f"completion_mask length ({len(completion_mask)}) != "
        f"completion_ids length ({len(completion_ids)})"
    )
    assert len(logprobs) == len(completion_ids), (
        f"logprobs length ({len(logprobs)}) != "
        f"completion_ids length ({len(completion_ids)})"
    )

    return prompt_ids, completion_ids, completion_mask, logprobs


async def process_sample(
    client: httpx.AsyncClient,
    prompt_data: dict,
    eos_token: str,
    server_url: str,
    compute_reward_fn: Callable[[str, Sample, str], RewardResult],
    tokenizer: Any = None,
    timing_out: dict | None = None,
    on_generation_complete: Callable[[], None] | None = None,
    lifecycle: SampleLifecycleCallbacks | None = None,
    env_worker_pool: Any | None = None,
) -> dict:
    """
    Process a single prompt and compute rewards.

    Args:
        client: HTTP client
        prompt_data: Dict with 'prompt' (raw question), 'sample', 'env_name', 'env'
        eos_token: EOS token for reward computation
        server_url: Inference server URL
        compute_reward_fn: Function to compute rewards (from scheduler/environment)
        tokenizer: HuggingFace tokenizer for formatting prompts with chat template
        on_generation_complete: Optional callback invoked after generation succeeds
            but before compute_reward starts (used to free lane slots early).
        lifecycle: Optional per-sample lifecycle callbacks for decoupled event logging.

    Returns:
        Dict with completion data, rewards, timing info.
        If there's an error, returns dict with "error" key set.
    """
    raw_prompt = prompt_data["prompt"]
    sample = prompt_data["sample"]
    env = prompt_data.get("env")
    
    prepared_prompt = prompt_data.get("prepared_prompt")
    prepared_prompt_token_count = prompt_data.get("prepared_prompt_token_count")
    
    # Format prompt using environment's format_prompt with proper tokenizer
    # This ensures the chat template is applied correctly
    if prepared_prompt is not None:
        prompt = prepared_prompt
        prompt_token_count = prepared_prompt_token_count
    elif env is not None and hasattr(env, 'format_prompt'):
        chat_kwargs = get_chat_template_kwargs()
        prompt = env.format_prompt(raw_prompt, tokenizer=tokenizer, chat_template_kwargs=chat_kwargs)
        prompt_token_count = None
    else:
        # Multi-turn environments don't implement format_prompt; use raw prompt
        prompt = raw_prompt
        prompt_token_count = None

    if lifecycle is not None and lifecycle.on_generation_start is not None:
        lifecycle.on_generation_start(0)

    try:
        completion_data, start_time, end_time, req_max_tokens = await generate_completion(
            client,
            prompt,
            server_url,
            prompt_token_count=prompt_token_count,
            timing_out=timing_out,
        )
    except PromptTooLongError as e:
        # Return error result - the group processor will handle this
        return {
            "error": "prompt_too_long",
            "error_message": str(e),
            "prompt_tokens": e.prompt_tokens,
            "max_tokens": e.max_tokens,
        }
    except RolloutError as e:
        return {
            "error": "rollout_error",
            "error_message": str(e),
            "error_type": e.error_type,
        }

    choice = completion_data["choices"][0]
    completion_text = choice["text"]

    if lifecycle is not None and lifecycle.on_generation_end is not None:
        lifecycle.on_generation_end(0)
    if on_generation_complete is not None:
        on_generation_complete()
    if lifecycle is not None and lifecycle.on_inference_end is not None:
        lifecycle.on_inference_end()

    if lifecycle is not None and lifecycle.on_reward_start is not None:
        lifecycle.on_reward_start()
    _reward_t0 = time.monotonic()
    if env_worker_pool is not None:
        env_name = prompt_data.get("env_name", "")
        reward_result = await env_worker_pool.compute_reward(
            env_name=env_name,
            is_multi_turn=False,
            completion_or_state=completion_text,
            sample=sample,
            eos_token=eos_token,
        )
    else:
        reward_result = await compute_reward_fn(completion_text, sample, eos_token)
    compute_reward_time = time.monotonic() - _reward_t0
    if lifecycle is not None and lifecycle.on_reward_end is not None:
        lifecycle.on_reward_end(compute_reward_time)
    total_reward = reward_result.total_reward
    sample_metrics = reward_result.sample_metrics
    golden_answers = reward_result.golden_answers
    sample_tags = reward_result.sample_tags

    # Extract token counts from usage
    usage = completion_data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Extract logprobs from vLLM response (for TIS/PPO clipping)
    # vLLM returns logprobs in choices[0]["logprobs"]["token_logprobs"]
    logprobs_data = choice.get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])
    # token_logprobs is a list of logprobs for each generated token
    # First token might be None in some vLLM versions, handle that
    vllm_logprobs = [lp if lp is not None else 0.0 for lp in token_logprobs]

    # Capture vLLM request ID for tracing correlation
    # This is the CompletionResponse.id (e.g. "cmpl-abc123")
    vllm_request_id = completion_data.get("id", "")

    info_turns = reward_result.info_turns

    return {
        "data_completion": completion_data,
        "reward": total_reward,
        "sample_metrics": sample_metrics,  # Per-sample metrics (reward components + other metrics)
        "golden_answers": golden_answers,  # Golden answer per reward component
        "info_turns": info_turns,  # Per-turn text info (e.g. stderr, summaries)
        "sample_tags": sample_tags,  # Per-sample string tags for filtering
        "request_timing": {
            "start_time": start_time,
            "end_time": end_time,
            "prompt_tokens": prompt_tokens,
            "rollout_tokens": completion_tokens,
            "vllm_request_id": vllm_request_id,
            "max_tokens": req_max_tokens,
            "compute_reward_time": compute_reward_time,
        },
        "vllm_logprobs": vllm_logprobs,
        "compute_reward_time": compute_reward_time,
        # Generation-centric data (single generation for single-turn)
        "generations": [{
            "generation_idx": 0,
            "content": completion_text,
            "tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tool_call_count": 0,
            "stop_reason": choice.get("finish_reason", ""),
            # vLLM timing from request (populated via OTLP tracing in orchestrator)
            # These get populated later in _log_kept_rollout_group from request_timing
        }],
        "env_responses": [],
        "tool_calls": [],
    }


async def process_multiturn_sample(
    client: httpx.AsyncClient,
    env: MultiTurnEnvironment,
    sample: Sample,
    eos_token: str,
    server_url: str,
    tokenizer: Any = None,
    prompt_data: dict | None = None,
    timing_out: dict | None = None,
    on_generation_complete: Callable[[], None] | None = None,
    lifecycle: SampleLifecycleCallbacks | None = None,
    env_worker_pool: Any | None = None,
) -> dict:
    """
    Process a single multi-turn sample.

    Runs a complete rollout and computes the reward.

    Args:
        on_generation_complete: Optional callback invoked after rollout succeeds
            but before compute_reward starts (used to free lane slots early).
        lifecycle: Optional per-sample lifecycle callbacks for decoupled event logging.

    Returns:
        Dict with trajectory data, rewards, timing info.
        If there's an error, returns dict with "error" key set.
    """
    try:
        prefetched_messages = None
        prefetched_prompt_str = None
        prefetched_prompt_token_count = None
        if prompt_data:
            prefetched_messages = prompt_data.get("prefetched_messages")
            prefetched_prompt_str = prompt_data.get("prefetched_prompt_str")
            prefetched_prompt_token_count = prompt_data.get("prefetched_prompt_token_count")
        
        state, request_timings = await run_multiturn_rollout(
            client,
            env,
            sample,
            server_url,
            tokenizer,
            prefetched_messages=prefetched_messages,
            prefetched_prompt_str=prefetched_prompt_str,
            prefetched_prompt_token_count=prefetched_prompt_token_count,
            timing_out=timing_out,
            lifecycle=lifecycle,
        )
    except Exception as e:
        return {
            "error": "rollout_error",
            "error_message": str(e),
        }

    if on_generation_complete is not None:
        on_generation_complete()
    if lifecycle is not None and lifecycle.on_inference_end is not None:
        lifecycle.on_inference_end()

    # Check for errors in state (but context_exhausted is OK)
    if state.error:
        return {
            "error": "rollout_error",
            "error_message": state.error,
        }

    # Need at least one turn for valid training data
    if state.num_turns == 0:
        return {
            "error": "empty_trajectory",
            "error_message": "No turns completed in rollout",
        }

    if lifecycle is not None and lifecycle.on_reward_start is not None:
        lifecycle.on_reward_start()
    _reward_t0 = time.monotonic()
    if env_worker_pool is not None:
        reward_result = await env_worker_pool.compute_reward(
            env_name=env.name,
            is_multi_turn=True,
            completion_or_state=state,
            sample=state.sample,
            eos_token=eos_token,
        )
    else:
        reward_result = await env.compute_reward(state, eos_token)
    compute_reward_time = time.monotonic() - _reward_t0
    if lifecycle is not None and lifecycle.on_reward_end is not None:
        lifecycle.on_reward_end(compute_reward_time)

    # Build interleaved completion data.
    # The completion_ids include env response tokens between model completions,
    # and the completion_mask marks which tokens the model generated (1) vs
    # env response tokens that should be masked out during training (0).
    prompt_ids, completion_ids, completion_mask, vllm_logprobs = _interleave_trajectory(state)
    
    # Get completion text by joining all assistant messages
    completion_text = ""
    for step in state.trajectory:
        if isinstance(step.completion, list):
            for msg in step.completion:
                if msg.get("role") == "assistant":
                    completion_text += msg.get("content", "") + "\n"
        elif isinstance(step.completion, str):
            completion_text += step.completion + "\n"
    completion_text = completion_text.strip()
    
    # Get prompt text and system prompt from the actual trajectory messages
    prompt_text = ""
    system_prompt = ""
    if state.trajectory and isinstance(state.trajectory[0].prompt, list):
        for msg in state.trajectory[0].prompt:
            if msg.get("role") == "user" and not prompt_text:
                prompt_text = msg.get("content", "")
            elif msg.get("role") == "system" and not system_prompt:
                system_prompt = msg.get("content", "")
    
    # Get logged data from state
    logged_turns = state.custom.get("_logged_turns", [])
    logged_generations = state.custom.get("_logged_generations", [])
    logged_env_responses = state.custom.get("_logged_env_responses", [])
    logged_tool_calls = state.custom.get("_logged_tool_calls", [])
    
    # Get full token sequence for raw_string (includes env responses)
    full_token_ids = state.custom.get("_full_token_ids", [])
    # Fallback: if full_token_ids not available, use prompt + completion
    if not full_token_ids:
        full_token_ids = prompt_ids + completion_ids
    
    return {
        "state": state,
        "reward": reward_result.total_reward,
        "sample_metrics": reward_result.sample_metrics,
        "golden_answers": reward_result.golden_answers,
        "info_turns": reward_result.info_turns,  # Per-turn text info (e.g. stderr, summaries)
        "sample_tags": reward_result.sample_tags,  # Per-sample string tags for filtering
        "request_timings": request_timings,
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,  # 1 for model tokens, 0 for env response tokens
        "full_token_ids": full_token_ids,  # Full sequence for raw_string
        "vllm_logprobs": vllm_logprobs,
        "completion_text": completion_text,
        "prompt_text": prompt_text,
        "system_prompt": system_prompt,
        "num_turns": state.num_turns,
        "stop_reason": state.stop_reason,
        "turns": logged_turns,
        "generations": logged_generations,
        "env_responses": logged_env_responses,
        "tool_calls": logged_tool_calls,
        "compute_reward_time": compute_reward_time,
    }


async def process_multiturn_group(
    prompt_data: dict,
    eos_token: str,
    server_url: str,
    env: MultiTurnEnvironment,
    tokenizer: Any = None,
    http_client: httpx.AsyncClient = None,
    sample_timings: list[dict] | None = None,
    sample_lifecycles: list[SampleLifecycleCallbacks] | None = None,
    env_worker_pool: Any | None = None,
) -> dict:
    """
    Process a group of multi-turn samples.

    Each sample runs an independent rollout with the same initial prompt.

    Args:
        prompt_data: Dict with 'prompt', 'sample', 'env_name'
        eos_token: EOS token for reward computation
        server_url: Inference server URL
        env: Multi-turn environment instance
        tokenizer: Optional tokenizer for formatting
        http_client: Optional HTTP client (if None, creates one)
        sample_lifecycles: Optional per-sample lifecycle callbacks.

    Returns:
        Dict with all rollout data, rewards, advantages, request timings.
    """
    sample = prompt_data["sample"]

    # Use provided client or create a new one
    if http_client is not None:
        tasks = [
            process_multiturn_sample(
                http_client,
                env,
                sample,
                eos_token,
                server_url,
                tokenizer,
                prompt_data=prompt_data,
                timing_out=sample_timings[i] if sample_timings else None,
                lifecycle=sample_lifecycles[i] if sample_lifecycles else None,
                env_worker_pool=env_worker_pool,
            )
            for i in range(config.cfg.group_size)
        ]
        group_samples = await asyncio.gather(*tasks)
    else:
        limits = httpx.Limits(max_connections=8192, max_keepalive_connections=0)
        async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(1200.0)) as client:
            tasks = [
                process_multiturn_sample(
                    client,
                    env,
                    sample,
                    eos_token,
                    server_url,
                    tokenizer,
                    prompt_data=prompt_data,
                    timing_out=sample_timings[i] if sample_timings else None,
                    lifecycle=sample_lifecycles[i] if sample_lifecycles else None,
                    env_worker_pool=env_worker_pool,
                )
                for i in range(config.cfg.group_size)
            ]
            group_samples = await asyncio.gather(*tasks)

    # Check if any sample had an error
    errors = [s for s in group_samples if "error" in s]
    if errors:
        error_sample = errors[0]
        prompt_text = ""
        if isinstance(sample.prompt, str):
            prompt_text = sample.prompt
        elif isinstance(sample.prompt, list) and sample.prompt:
            prompt_text = sample.prompt[0].get("content", "")
        
        return {
            "error": error_sample["error"],
            "error_message": error_sample.get("error_message", "Unknown error"),
            "prompt_text": prompt_text,
            "env_name": prompt_data["env_name"],
            "server_url": server_url,
        }

    # Extract data
    rewards = [s["reward"] for s in group_samples]
    sample_metrics_list = [s["sample_metrics"] for s in group_samples]
    golden_answers_list = [s["golden_answers"] for s in group_samples]
    info_turns_list = [s["info_turns"] for s in group_samples]
    sample_tags_list = [s["sample_tags"] for s in group_samples]
    completion_texts = [s["completion_text"] for s in group_samples]
    prompt_texts = [s["prompt_text"] for s in group_samples]
    compute_reward_times = [s["compute_reward_time"] for s in group_samples]

    # Flatten all request timings with sample index metadata
    all_request_timings = []
    for sample_idx, s in enumerate(group_samples):
        sample_crt = s.get("compute_reward_time", 0.0)
        for timing in s["request_timings"]:
            timing_with_sample = dict(timing)
            timing_with_sample["sample_idx_in_group"] = sample_idx
            timing_with_sample["compute_reward_time"] = sample_crt
            all_request_timings.append(timing_with_sample)
    
    # Extract token data
    prompt_token_ids = [s["prompt_ids"] for s in group_samples]
    completion_token_ids = [s["completion_ids"] for s in group_samples]
    completion_masks = [s["completion_mask"] for s in group_samples]  # 1=model, 0=env response
    full_token_ids = [s["full_token_ids"] for s in group_samples]  # Full sequence for raw_string
    vllm_logprobs = [s["vllm_logprobs"] for s in group_samples]
    
    # Compute total tokens
    total_tokens = [
        len(s["prompt_ids"]) + len(s["completion_ids"]) 
        for s in group_samples
    ]

    # --- Overlong soft penalty: graduated reward penalty as response approaches max_tokens ---
    # For multi-turn, use model-generated tokens only (exclude env response tokens).
    if config.cfg.overlong_penalty_factor > 0:
        max_resp_len = config.cfg.max_tokens
        buffer = config.cfg.overlong_buffer_tokens
        for i, s in enumerate(group_samples):
            resp_len = sum(s["completion_mask"])  # count model tokens only (1s in mask)
            exceed = resp_len - (max_resp_len - buffer)
            if exceed > 0:
                penalty = min(-exceed / buffer * config.cfg.overlong_penalty_factor, 0.0)
                rewards[i] += penalty

    # Compute advantages (standardized rewards)
    advantages = compute_advantages(rewards)

    # Extract turns for each sample
    turns_list = [s["turns"] for s in group_samples]
    
    # Extract system_prompt and user prompt (raw text, no chat template)
    # Prefer per-sample system_prompt (extracted from trajectory messages),
    # fall back to env-level system_prompt for envs that use a class-level one.
    system_prompt = group_samples[0].get("system_prompt", "") if group_samples else ""
    if not system_prompt:
        system_prompt = env.system_prompt or ""
    
    # Use prompt_texts which has raw user message content
    user_prompt = prompt_texts[0] if prompt_texts else ""
    
    # Compute token counts
    tokens_system_prompt = 0
    if system_prompt and tokenizer is not None:
        tokens_system_prompt = len(tokenizer.encode(system_prompt))
    
    return {
        "prompt_text": user_prompt,  # Raw user prompt (no chat template)
        "env_name": prompt_data["env_name"],
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "completion_masks": completion_masks,  # Per-sample: 1 for model tokens, 0 for env response tokens
        "full_token_ids": full_token_ids,  # Full sequence for raw_string (includes env responses)
        "completion_texts": completion_texts,
        "total_tokens": total_tokens,
        "rewards": rewards,
        "advantages": advantages,
        "sample_metrics": sample_metrics_list,
        "golden_answers": golden_answers_list,
        "info_turns": info_turns_list,  # Per-sample list of per-turn text info dicts
        "sample_tags": sample_tags_list,  # Per-sample string tags for filtering
        "request_timings": all_request_timings,
        "vllm_logprobs": vllm_logprobs,
        "server_url": server_url,
        "is_multiturn": True,
        "num_turns": [s["num_turns"] for s in group_samples],
        "stop_reasons": [s["stop_reason"] for s in group_samples],
        "turns": turns_list,
        "generations": [s["generations"] for s in group_samples],
        "env_responses": [s["env_responses"] for s in group_samples],
        "tool_calls": [s["tool_calls"] for s in group_samples],
        "system_prompt": system_prompt,  # Raw system prompt (no chat template)
        "tokens_system_prompt": tokens_system_prompt,
        "compute_reward_times": compute_reward_times,
        "is_truncated": [s["stop_reason"] == "context_exhausted" for s in group_samples],
    }


async def process_group(
    prompt_data: dict,
    eos_token: str,
    server_url: str,
    compute_reward_fn: Callable[[str, Sample, str], RewardResult],
    env: MultiTurnEnvironment | None = None,
    tokenizer: Any = None,
    http_client: httpx.AsyncClient = None,
    sample_timings: list[dict] | None = None,
    sample_lifecycles: list[SampleLifecycleCallbacks] | None = None,
    env_worker_pool: Any | None = None,
) -> dict:
    """
    Process a group of samples (same prompt, multiple completions).

    Automatically handles both single-turn and multi-turn environments.

    Args:
        prompt_data: Dict with 'prompt' (raw question), 'sample', 'env_name', 'env'
        eos_token: EOS token for reward computation
        server_url: Inference server URL
        compute_reward_fn: Function to compute rewards (for single-turn)
        env: Optional multi-turn environment instance
        tokenizer: Tokenizer for chat template formatting (required for single-turn)
        http_client: Optional HTTP client (if None, creates one)
        sample_lifecycles: Optional per-sample lifecycle callbacks.

    Returns:
        Dict with all completion data, rewards, advantages, request timings, and vllm logprobs.
        If the prompt is too long, returns dict with "error" key set.

    Note:
        prompt_data['prompt'] should be the raw question text. The chat template
        formatting is applied at rollout time using the environment's format_prompt()
        method with the provided tokenizer.
    """
    # Check if this is a multi-turn environment
    if env is not None and getattr(env, 'is_multi_turn', False):
        return await process_multiturn_group(
            prompt_data, eos_token, server_url, env, tokenizer, http_client,
            sample_timings=sample_timings,
            sample_lifecycles=sample_lifecycles,
            env_worker_pool=env_worker_pool,
        )

    # Single-turn processing - use provided client or create a new one
    if http_client is not None:
        tasks = [
            process_sample(http_client, prompt_data, eos_token, server_url, compute_reward_fn, tokenizer,
                           timing_out=sample_timings[i] if sample_timings else None,
                           lifecycle=sample_lifecycles[i] if sample_lifecycles else None,
                           env_worker_pool=env_worker_pool)
            for i in range(config.cfg.group_size)
        ]
        group_samples = await asyncio.gather(*tasks)
    else:
        limits = httpx.Limits(max_connections=8192, max_keepalive_connections=0)
        async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(1200.0)) as client:
            tasks = [
                process_sample(client, prompt_data, eos_token, server_url, compute_reward_fn, tokenizer,
                               timing_out=sample_timings[i] if sample_timings else None,
                               lifecycle=sample_lifecycles[i] if sample_lifecycles else None,
                               env_worker_pool=env_worker_pool)
                for i in range(config.cfg.group_size)
            ]
            group_samples = await asyncio.gather(*tasks)

    # Check if any sample had an error (they all use the same prompt, so if one fails, all fail)
    errors = [s for s in group_samples if "error" in s]
    if errors:
        error_sample = errors[0]
        return {
            "error": error_sample["error"],
            "error_message": error_sample.get("error_message", "Unknown error"),
            "prompt_text": prompt_data["prompt"],
            "env_name": prompt_data["env_name"],
            "server_url": server_url,
            # Include token info if available (for prompt_too_long errors)
            "prompt_tokens": error_sample.get("prompt_tokens"),
            "max_tokens": error_sample.get("max_tokens"),
        }

    # Extract data
    rewards = [s["reward"] for s in group_samples]
    sample_metrics_list = [s["sample_metrics"] for s in group_samples]
    golden_answers_list = [s["golden_answers"] for s in group_samples]
    info_turns_list = [s["info_turns"] for s in group_samples]
    sample_tags_list = [s["sample_tags"] for s in group_samples]
    compute_reward_times = [s["compute_reward_time"] for s in group_samples]
    completion_texts = [
        s["data_completion"]["choices"][0].get("text", "")
        for s in group_samples
    ]
    
    # Extract request timings with sample index metadata
    request_timings = []
    for sample_idx, s in enumerate(group_samples):
        timing_with_sample = dict(s["request_timing"])
        timing_with_sample["sample_idx_in_group"] = sample_idx
        request_timings.append(timing_with_sample)
    
    # Extract vLLM logprobs for TIS/PPO clipping
    vllm_logprobs = [s["vllm_logprobs"] for s in group_samples]

    # Per-sample truncation flags (finish_reason == "length" means hit max_tokens without EOS)
    is_truncated_list = [
        s["data_completion"]["choices"][0].get("finish_reason", "") == "length"
        for s in group_samples
    ]

    # --- Overlong soft penalty: graduated reward penalty as response approaches max_tokens ---
    if config.cfg.overlong_penalty_factor > 0:
        max_resp_len = config.cfg.max_tokens
        buffer = config.cfg.overlong_buffer_tokens
        for i, s in enumerate(group_samples):
            resp_len = s["data_completion"]["usage"].get("completion_tokens", 0)
            exceed = resp_len - (max_resp_len - buffer)
            if exceed > 0:
                penalty = min(-exceed / buffer * config.cfg.overlong_penalty_factor, 0.0)
                rewards[i] += penalty

    # Compute advantages (standardized rewards)
    advantages = compute_advantages(rewards)

    # Create turns data for single-turn completions (one turn per sample)
    # Include tokens from request_timing and stop_reason (vLLM finish_reason)
    turns_list = [
        [{
            "turn_order": 0,
            "turn_type": "model",
            "content": completion_text,
            "tokens": request_timings[i].get("rollout_tokens", 0),
            "stop_reason": group_samples[i]["data_completion"]["choices"][0].get("finish_reason", ""),
        }]
        for i, completion_text in enumerate(completion_texts)
    ]

    # Extract system_prompt and user prompt from the environment/sample (raw text, no chat template)
    env = prompt_data.get("env")
    system_prompt = (env.system_prompt or "") if env else ""
    instruction_prompt = (getattr(env, "instruction_prompt", None) or "") if env else ""
    
    # Build the full user content (instruction + question) like format_prompt does
    # sample.prompt is the raw question, but actual user message includes instruction_prompt
    sample = prompt_data.get("sample")
    raw_question = ""
    if sample:
        raw_question = sample.prompt if isinstance(sample.prompt, str) else sample.metadata.get("question", "")
    
    # Combine instruction_prompt + question (same as format_prompt does)
    if instruction_prompt and raw_question:
        user_prompt = f"{instruction_prompt}\n\n{raw_question}"
    else:
        user_prompt = raw_question
    
    # Compute token counts
    tokens_system_prompt = 0
    if system_prompt and tokenizer is not None:
        tokens_system_prompt = len(tokenizer.encode(system_prompt))

    return {
        "prompt_text": user_prompt,  # Raw user prompt (no chat template)
        "env_name": prompt_data["env_name"],
        "prompt_token_ids": [s["data_completion"]["choices"][0]["prompt_token_ids"] for s in group_samples],
        "completion_token_ids": [s["data_completion"]["choices"][0]["token_ids"] for s in group_samples],
        "completion_texts": completion_texts,
        "total_tokens": [s["data_completion"]["usage"]["total_tokens"] for s in group_samples],
        "rewards": rewards,
        "advantages": advantages,
        "sample_metrics": sample_metrics_list,  # List of dicts, per-sample metrics (reward components + other metrics)
        "golden_answers": golden_answers_list,  # List of dicts mapping golden answer keys -> values
        "info_turns": info_turns_list,  # Per-sample list of per-turn text info dicts
        "sample_tags": sample_tags_list,  # Per-sample string tags for filtering
        "request_timings": request_timings,
        "vllm_logprobs": vllm_logprobs,
        "server_url": server_url,
        "turns": turns_list,
        "generations": [s["generations"] for s in group_samples],
        "env_responses": [s["env_responses"] for s in group_samples],
        "tool_calls": [s["tool_calls"] for s in group_samples],
        "system_prompt": system_prompt,  # Raw system prompt (no chat template)
        "tokens_system_prompt": tokens_system_prompt,
        "compute_reward_times": compute_reward_times,
        "is_truncated": is_truncated_list,
    }


def compute_advantages(rewards: list[float]) -> list[float]:
    """Compute per-sample advantages from a prompt group's rewards.

    Strategy varies by config.cfg.algorithm and config.cfg.advantage_norm.
    """
    import numpy as np
    rewards_np = np.array(rewards)
    n = len(rewards_np)

    algo = config.cfg.algorithm

    # --- RLOO: leave-one-out baseline (always per-group) ---
    if algo == "rloo":
        mean = rewards_np.mean()
        if n > 1:
            return ((rewards_np - mean) * n / (n - 1)).tolist()
        return (rewards_np * 0.0).tolist()

    # --- Dr.GRPO: mean-centered, NO std division ---
    if algo == "dr_grpo":
        return (rewards_np - rewards_np.mean()).tolist()

    # --- Batch-level normalization (REINFORCE++ or explicit advantage_norm="batch") ---
    # Two-stage process following REINFORCE++ arXiv:2501.03262 §3.2:
    #   Stage 1 (here): subtract group mean to remove prompt-difficulty bias.
    #   Stage 2 (batch_processor): whiten across full batch (mean + std).
    # For k=1 (single sample per group) group mean = the reward itself,
    # so subtraction would zero everything out — pass raw rewards instead.
    if algo == "reinforce_pp" or config.cfg.advantage_norm == "batch":
        if n > 1:
            return (rewards_np - rewards_np.mean()).tolist()
        return rewards_np.tolist()

    # --- Default group normalization (grpo, cispo, gspo, sapo) ---
    std = rewards_np.std(ddof=1) if n > 1 else 0.0
    return ((rewards_np - rewards_np.mean()) / (std + 1e-4)).tolist()

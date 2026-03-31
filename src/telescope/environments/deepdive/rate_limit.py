from __future__ import annotations

import asyncio
import functools
import random

from openai import RateLimitError


def with_rate_limit_retry(
    concurrency_semaphore: asyncio.Semaphore,
    delay_semaphore: asyncio.Semaphore,
    rate_limit_event: asyncio.Event,
    max_retries: int = 5,
    base_delay: float = 1.0,
):
    """
    Decorator for async functions to handle OpenAI-style rate limiting with
    shared backoff coordination across concurrent tasks.

    Uses a shared Event to temporarily pause new calls when any call hits a 429.
    Backoff curve ~ 1.36787944**attempt with jitter (constant ~= 1 + 1/e).
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    async with concurrency_semaphore:
                        await rate_limit_event.wait()
                        if attempt > 0:
                            await asyncio.sleep(random.uniform(0, 2))

                        return await func(*args, **kwargs)

                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise

                    rate_limit_event.clear()

                    delay = base_delay * (1.36787944**attempt) + random.uniform(0, 1)

                    async with delay_semaphore:
                        await asyncio.sleep(delay)
                        rate_limit_event.set()

        return wrapper

    return decorator

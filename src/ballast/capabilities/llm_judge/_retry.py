"""Exponential-backoff retry helper for judge model calls.

Standalone (no LLMJudge / pydantic-evals coupling) so the same
retry shape can be reused if other judge-like primitives appear.
Translates the final exception into :class:`JudgeUnavailable` so the
caller's try/except can distinguish infrastructure failure from a
real low-score verdict.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ballast.capabilities.llm_judge._errors import JudgeUnavailable

R = TypeVar("R")


async def retry_with_backoff(
    call: Callable[[], Awaitable[R]],
    *,
    max_retries: int,
    backoff_base_s: float,
    model_id: str,
) -> R:
    """Execute ``call`` with up to ``max_retries`` retries on any
    exception. Backoff doubles each attempt starting from
    ``backoff_base_s`` (so retries 1..N sleep ``base, 2*base, 4*base, …``).

    On exhaustion raises :class:`JudgeUnavailable` with the original
    exception attached as ``last_error``.
    """
    attempts = max_retries + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 — wrap all transients
            last_error = exc
            if attempt + 1 >= attempts:
                break
            await asyncio.sleep(backoff_base_s * (2 ** attempt))
    assert last_error is not None  # loop ran at least once
    raise JudgeUnavailable(
        attempts=attempts, last_error=last_error, model_used=model_id,
    )


__all__ = ["retry_with_backoff"]

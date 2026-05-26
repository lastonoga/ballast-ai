"""``as_workflow_decorator`` — workflow surface for CircuitBreaker."""
from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker

T = TypeVar("T")


def as_workflow_decorator(
    breaker: CircuitBreaker, *,
    scope_ctx: Mapping[str, Any] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorate an async function so its invocations flow through the breaker.

    ``scope_ctx`` is forwarded to ``breaker.scope_key`` to determine the
    breaker's per-scope bucket. None → ``scope_key`` receives ``{}``.
    """
    ctx = scope_ctx or {}

    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.call(fn, *args, ctx=ctx, **kwargs)

        return wrapper

    return deco


__all__ = ["as_workflow_decorator"]

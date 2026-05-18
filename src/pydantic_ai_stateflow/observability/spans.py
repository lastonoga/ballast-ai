from __future__ import annotations

import functools
import importlib
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

AttrsFn = Callable[..., dict[str, Any]]


def _get_logfire_span() -> Callable[..., Any] | None:
    try:
        mod = importlib.import_module("logfire")
        if mod is None:
            return None
        span = getattr(mod, "span", None)
        return span if callable(span) else None
    except Exception:
        return None


def traced(
    name: str, *, attrs: AttrsFn | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Wrap an async function in a logfire span.

    No-op when logfire is missing — the wrapped function runs unchanged.
    `attrs` callable receives the wrapped function's args and returns a
    dict merged into the span attributes (canonical names per spec 4D).
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            span_fn = _get_logfire_span()
            if span_fn is None:
                return await fn(*args, **kwargs)
            try:
                attributes = attrs(*args, **kwargs) if attrs else {}
            except Exception:
                attributes = {}
            with span_fn(name, **attributes):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator

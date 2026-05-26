"""as_workflow_decorator — decorates an async fn, runs it through CircuitBreaker."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.resilience.circuit_breaker._adapters.workflow import as_workflow_decorator
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import ReturnValue
from ballast.resilience.circuit_breaker._scope import per_tool_scope
from ballast.resilience.circuit_breaker._state import BreakerState
from ballast.resilience.circuit_breaker._thresholds import Consecutive


class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


@pytest.mark.asyncio
async def test_decorator_passes_through_when_closed() -> None:
    cb = CircuitBreaker(clock=_Clock())

    @as_workflow_decorator(cb)
    async def body(x: int) -> int:
        return x * 2

    assert await body(5) == 10


@pytest.mark.asyncio
async def test_decorator_opens_after_failures_then_uses_fallback() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        fallback=ReturnValue("fallback"),
        clock=_Clock(),
    )

    @as_workflow_decorator(cb)
    async def body() -> str:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await body()
    with pytest.raises(RuntimeError):
        await body()
    assert await body() == "fallback"


@pytest.mark.asyncio
async def test_decorator_propagates_scope_ctx() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        scope_key=per_tool_scope,
        clock=_Clock(),
    )

    @as_workflow_decorator(cb, scope_ctx={"tool_name": "publish_wf"})
    async def body() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await body()
    assert cb.stats("tool:publish_wf").state == BreakerState.OPEN

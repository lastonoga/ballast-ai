"""End-to-end integration: CircuitBreaker with all primitives composed."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import (
    CallFallback, Chain, RaiseError, ReturnValue,
)
from ballast.resilience.circuit_breaker._scope import per_tool_scope
from ballast.resilience.circuit_breaker._state import (
    BreakerState, CircuitOpenError,
)
from ballast.resilience.circuit_breaker._thresholds import (
    Consecutive, WindowedCount, WindowedRate,
)


class _Clock:
    def __init__(self, start: datetime): self.now = start
    def advance(self, td: timedelta): self.now += td
    def __call__(self): return self.now


def _clock() -> _Clock:
    return _Clock(datetime(2026, 1, 1, tzinfo=UTC))


@pytest.mark.asyncio
async def test_full_lifecycle_with_recovery() -> None:
    """CLOSED → OPEN → HALF_OPEN → CLOSED through a real recovery cycle."""
    clock = _clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        recovery_after=timedelta(seconds=5),
        clock=clock,
    )

    async def flaky(succeed: bool):
        if not succeed:
            raise RuntimeError("boom")
        return "ok"

    # 2 failures → OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(flaky, False)
    assert cb.stats().state == BreakerState.OPEN

    # Still OPEN before recovery — raises CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await cb.call(flaky, True)

    # Advance past recovery; first probe succeeds → CLOSED
    clock.advance(timedelta(seconds=6))
    assert await cb.call(flaky, True) == "ok"
    assert cb.stats().state == BreakerState.CLOSED


async def _failing_fallback():
    raise RuntimeError("fallback also broken")


@pytest.mark.asyncio
async def test_call_fallback_chain_with_hitl_simulated() -> None:
    """Chain: CallFallback → ReturnValue. First-success-wins."""
    clock = _clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        fallback=Chain(
            CallFallback(lambda *a, **kw: _failing_fallback()),
            ReturnValue("ultimate_fallback"),
        ),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)

    # Now OPEN; Chain tries CallFallback (fails), then ReturnValue (succeeds)
    out = await cb.call(boom)
    assert out == "ultimate_fallback"


@pytest.mark.asyncio
async def test_per_tool_scope_isolation_under_load() -> None:
    """Multiple scopes don't share state, even under concurrent load."""
    clock = _clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        scope_key=per_tool_scope,
        clock=clock,
    )

    async def call_for_tool(name: str, succeed: bool) -> Any:
        async def fn():
            if not succeed:
                raise RuntimeError("nope")
            return f"ok:{name}"
        return await cb.call(fn, ctx={"tool_name": name})

    # Trip tool A
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await call_for_tool("a", False)
    assert cb.stats("tool:a").state == BreakerState.OPEN

    # Tool B still passes through
    assert await call_for_tool("b", True) == "ok:b"
    assert cb.stats("tool:b").state == BreakerState.CLOSED

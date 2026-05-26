"""CircuitBreaker core — .call(), per-scope state, transitions."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import RaiseError, ReturnValue
from ballast.resilience.circuit_breaker._scope import global_scope, per_tool_scope
from ballast.resilience.circuit_breaker._state import (
    BreakerState, CircuitOpenError,
)
from ballast.resilience.circuit_breaker._thresholds import Consecutive


# ---- Mockable clock --------------------------------------------------------

class _Clock:
    def __init__(self, start: datetime): self.now = start
    def advance(self, td: timedelta) -> None: self.now += td
    def __call__(self) -> datetime: return self.now


def _mk_clock() -> _Clock:
    return _Clock(datetime(2026, 1, 1, tzinfo=UTC))


# ---- Tests -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_closed_passes_through() -> None:
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(3), clock=_mk_clock())

    async def ok(): return "out"

    assert await cb.call(ok) == "out"
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_closed_to_open_after_threshold() -> None:
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(2), clock=_mk_clock())

    async def boom(): raise RuntimeError("nope")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_open_invokes_fallback_with_raise_error_default() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(1), clock=clock)

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)

    async def ok(): return "fresh"

    with pytest.raises(CircuitOpenError):
        await cb.call(ok)


@pytest.mark.asyncio
async def test_open_returns_via_return_value_fallback() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        fallback=ReturnValue("cached"),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)

    async def ok(): return "fresh"

    assert await cb.call(ok) == "cached"


@pytest.mark.asyncio
async def test_open_to_half_open_after_recovery() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        recovery_after=timedelta(seconds=10),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN

    clock.advance(timedelta(seconds=11))
    async def ok(): return "out"
    assert await cb.call(ok) == "out"
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_half_open_probe_failure_reopens() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        recovery_after=timedelta(seconds=10),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    clock.advance(timedelta(seconds=11))

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_half_open_extra_probes_rejected() -> None:
    clock = _mk_clock()
    rejected = []

    class _CapturingFallback:
        async def on_rejected(self, stats, fn, args, kwargs):
            rejected.append(stats.state)
            return "rejected"

    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        recovery_after=timedelta(seconds=10),
        probe_max=1,
        fallback=_CapturingFallback(),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")
    async def slow(): await asyncio.sleep(0.05); return "ok"

    with pytest.raises(RuntimeError):
        await cb.call(boom)

    clock.advance(timedelta(seconds=11))

    results = await asyncio.gather(
        cb.call(slow), cb.call(slow), cb.call(slow),
        return_exceptions=True,
    )
    assert "rejected" in results


@pytest.mark.asyncio
async def test_ignored_exception_does_not_count_as_failure() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        ignored_exc=(KeyError,),
        clock=clock,
    )

    async def boom(): raise KeyError("ignored")

    with pytest.raises(KeyError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_is_failure_exc_filters_other_exceptions() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        is_failure_exc=(RuntimeError,),
        clock=clock,
    )

    async def boom(): raise ValueError("other type")

    with pytest.raises(ValueError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_is_success_predicate_treats_returned_value_as_failure() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda r: r != "bad",
        clock=clock,
    )

    async def maybe_bad(): return "bad"

    await cb.call(maybe_bad)
    await cb.call(maybe_bad)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_per_scope_isolation() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        scope_key=per_tool_scope,
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")
    async def ok(): return "out"

    with pytest.raises(RuntimeError):
        await cb.call(boom, ctx={"tool_name": "search"})
    assert cb.stats("tool:search").state == BreakerState.OPEN

    assert await cb.call(ok, ctx={"tool_name": "other"}) == "out"
    assert cb.stats("tool:other").state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_reset_forces_closed() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(1), clock=clock)

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN

    cb.reset()
    assert cb.stats().state == BreakerState.CLOSED


def test_constructor_validates_probe_max() -> None:
    with pytest.raises(ValueError, match="probe_max"):
        CircuitBreaker(probe_max=0)

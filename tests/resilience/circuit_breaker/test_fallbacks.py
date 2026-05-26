"""Built-in FallbackPolicy implementations."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._fallbacks import (
    CallFallback, Chain, EscalateToHITL, RaiseError, ReturnValue,
)
from ballast.resilience.circuit_breaker._protocols import FallbackPolicy
from ballast.resilience.circuit_breaker._state import BreakerState, BreakerStats, CircuitOpenError


def _stats() -> BreakerStats:
    return BreakerStats(
        scope="x", state=BreakerState.OPEN,
        consecutive_failures=5, total_failures=5, total_successes=0,
        opened_at=None, will_attempt_recovery_at=None,
        probe_attempts=0, probe_max=1,
    )


async def _noop_fn(*args, **kwargs):
    return "real"


@pytest.mark.asyncio
async def test_raise_error_raises_circuit_open_error() -> None:
    with pytest.raises(CircuitOpenError) as exc:
        await RaiseError().on_rejected(_stats(), _noop_fn, (), {})
    assert exc.value.stats.scope == "x"


@pytest.mark.asyncio
async def test_return_value_returns_stored() -> None:
    fb = ReturnValue("cached")
    out = await fb.on_rejected(_stats(), _noop_fn, (), {})
    assert out == "cached"


@pytest.mark.asyncio
async def test_call_fallback_invokes_without_stats_param() -> None:
    seen: list[tuple] = []

    async def my_fb(a, b, *, c=None):
        seen.append((a, b, c))
        return "fb"

    out = await CallFallback(my_fb).on_rejected(_stats(), _noop_fn, (1, 2), {"c": "x"})
    assert out == "fb"
    assert seen == [(1, 2, "x")]


@pytest.mark.asyncio
async def test_call_fallback_invokes_with_stats_param() -> None:
    captured: dict = {}

    async def my_fb(a, *, stats=None):
        captured["a"] = a
        captured["stats"] = stats
        return "ok"

    out = await CallFallback(my_fb).on_rejected(_stats(), _noop_fn, (42,), {})
    assert out == "ok"
    assert captured["a"] == 42
    assert isinstance(captured["stats"], BreakerStats)


@pytest.mark.asyncio
async def test_escalate_to_hitl_calls_channel_request_blocking() -> None:
    requested = []

    class _Card:
        def __init__(self, stats): self.stats = stats

    class _FakeChannel:
        async def request(self, payload, *, timeout=None):
            requested.append({"payload": payload, "timeout": timeout})
            return "human_verdict"

    out = await EscalateToHITL(
        channel=_FakeChannel(),
        card_factory=_Card,
        timeout=timedelta(minutes=5),
    ).on_rejected(_stats(), _noop_fn, (), {})

    assert out == "human_verdict"
    assert len(requested) == 1
    assert isinstance(requested[0]["payload"], _Card)
    assert requested[0]["timeout"] == timedelta(minutes=5)


@pytest.mark.asyncio
async def test_chain_returns_first_success() -> None:
    class _Bad:
        async def on_rejected(self, *args): raise RuntimeError("nope")

    class _Ok:
        async def on_rejected(self, *args): return "from_ok"

    out = await Chain(_Bad(), _Ok()).on_rejected(_stats(), _noop_fn, (), {})
    assert out == "from_ok"


@pytest.mark.asyncio
async def test_chain_raises_last_when_all_fail() -> None:
    class _Bad1:
        async def on_rejected(self, *args): raise RuntimeError("first")

    class _Bad2:
        async def on_rejected(self, *args): raise ValueError("second")

    with pytest.raises(ValueError, match="second"):
        await Chain(_Bad1(), _Bad2()).on_rejected(_stats(), _noop_fn, (), {})


def test_chain_requires_at_least_one_policy() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Chain()

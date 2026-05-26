"""as_capability — wraps agent.run() through the breaker via after_run hook."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._adapters.capability import as_capability
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import RaiseError, ReturnValue
from ballast.resilience.circuit_breaker._state import BreakerState
from ballast.resilience.circuit_breaker._thresholds import Consecutive


class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


@pytest.mark.asyncio
async def test_as_capability_returns_ballast_capability_instance() -> None:
    from ballast.capabilities.base import BallastCapability
    cap = as_capability(CircuitBreaker(clock=_Clock()))
    assert isinstance(cap, BallastCapability)


@pytest.mark.asyncio
async def test_capability_records_failure_when_is_success_predicate_marks_bad() -> None:
    cb_with_pred = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda res: getattr(res, "output", None) != "bad",
        clock=_Clock(),
    )
    cap = as_capability(cb_with_pred)
    per_run = await cap.for_run(ctx=None)

    class _BadResult:
        output = "bad"

    await per_run.after_run(ctx=None, result=_BadResult())
    await per_run.after_run(ctx=None, result=_BadResult())
    # After two "bad" results, breaker should be OPEN
    assert cb_with_pred.stats().state == BreakerState.OPEN

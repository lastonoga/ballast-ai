"""BreakerStep — wraps a Step (PlanAndExecute) through the breaker."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.resilience.circuit_breaker._adapters.step import BreakerStep, as_step
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import ReturnValue
from ballast.resilience.circuit_breaker._scope import per_step_scope
from ballast.resilience.circuit_breaker._state import BreakerState
from ballast.resilience.circuit_breaker._thresholds import Consecutive


class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


class _OkStep:
    async def execute(self, plan_input, dep_outputs, ctx):
        return "ok"


class _BoomStep:
    async def execute(self, plan_input, dep_outputs, ctx):
        raise RuntimeError("step failed")


def _ctx(step_id: str = "s1", kind: str = "callable") -> StepContext:
    return StepContext(
        plan=Plan(steps=[PlannedStep(id=step_id, kind=kind)]),
        step=PlannedStep(id=step_id, kind=kind),
        step_registry=None,
    )


@pytest.mark.asyncio
async def test_breaker_step_passes_through_when_closed() -> None:
    cb = CircuitBreaker(clock=_Clock())
    wrapped = as_step(cb, _OkStep())
    out = await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx())
    assert out == "ok"


@pytest.mark.asyncio
async def test_breaker_step_uses_per_step_scope() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        scope_key=per_step_scope,
        clock=_Clock(),
    )
    wrapped = as_step(cb, _BoomStep())

    with pytest.raises(RuntimeError):
        await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx("s1"))
    assert cb.stats("step:s1").state == BreakerState.OPEN
    # Other step id still CLOSED
    assert cb.stats("step:s2").state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_breaker_step_routes_to_fallback_when_open() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        fallback=ReturnValue("fallback"),
        clock=_Clock(),
    )
    wrapped = as_step(cb, _BoomStep())

    with pytest.raises(RuntimeError):
        await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx())
    out = await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx())
    assert out == "fallback"


def test_breaker_step_constructor_via_as_step() -> None:
    cb = CircuitBreaker(clock=_Clock())
    bs = as_step(cb, _OkStep())
    assert isinstance(bs, BreakerStep)

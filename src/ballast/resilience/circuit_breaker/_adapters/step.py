"""``BreakerStep`` + ``as_step`` — wraps a PlanAndExecute Step through the breaker."""
from __future__ import annotations

from typing import Any

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker


class BreakerStep:
    """Wraps any ``Step`` (PlanAndExecute) — invocations flow through the breaker.

    The breaker's ``scope_key`` receives ``{"step_id": ctx.step.id, "step_kind": ctx.step.kind}``,
    so ``per_step_scope`` gives per-DAG-node isolation out of the box.
    """

    def __init__(self, *, breaker: CircuitBreaker, wrapped: Any) -> None:
        self._breaker = breaker
        self._wrapped = wrapped

    async def execute(self, plan_input, dep_outputs, ctx):
        scope_ctx = {"step_id": ctx.step.id, "step_kind": ctx.step.kind}
        return await self._breaker.call(
            self._wrapped.execute, plan_input, dep_outputs, ctx,
            ctx=scope_ctx,
        )


def as_step(breaker: CircuitBreaker, wrapped: Any) -> BreakerStep:
    """Wrap ``wrapped`` (any object implementing ``Step.execute``) through ``breaker``."""
    return BreakerStep(breaker=breaker, wrapped=wrapped)


__all__ = ["BreakerStep", "as_step"]

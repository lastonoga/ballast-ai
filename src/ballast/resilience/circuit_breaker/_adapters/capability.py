"""``as_capability`` — agent surface for CircuitBreaker.

Wraps full ``agent.run()`` invocations via the ``after_run`` hook.
Each agent run's result is fed through the breaker's ``is_success``
predicate; failures advance the counter.

Per-tool wrapping at the pydantic-ai tool level is OUT OF SCOPE for the
first cut — apps that need per-tool CB use the workflow decorator + a
manual ``breaker.call(...)`` wrapper around their tool function, OR
``BreakerStep`` for PlanAndExecute DAG nodes.
"""
from __future__ import annotations

import logging
from typing import Any

from ballast.capabilities.base import BallastCapability
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker

_log = logging.getLogger("ballast.resilience.circuit_breaker.capability")


class _CBCapability(BallastCapability):
    """Tracks agent.run() outcomes through the configured CircuitBreaker."""

    name = "circuit_breaker"

    def __init__(self, breaker: CircuitBreaker) -> None:
        self._breaker = breaker

    async def for_run(self, ctx: Any) -> "_CBCapability":
        # Stateless wrapper — same breaker shared across runs (it has its
        # own per-scope state already).
        return self

    async def after_run(self, ctx: Any, *, result: Any) -> Any:
        # Feed the run outcome through the breaker as an "invocation". We
        # use a synthetic no-op function so the breaker's bookkeeping
        # (success vs failure via is_success) updates correctly.
        async def _noop() -> Any:
            return result

        try:
            await self._breaker.call(_noop, ctx={"agent_run": True})
        except Exception:
            _log.exception("circuit-breaker after_run propagation swallowed")
        return result


def as_capability(breaker: CircuitBreaker) -> BallastCapability:
    """Wrap ``breaker`` as a ``BallastCapability`` for agent runs."""
    return _CBCapability(breaker)


__all__ = ["as_capability"]

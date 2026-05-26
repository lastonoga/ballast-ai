"""Built-in ``RePlanPolicy`` implementations.

First cut ships only ``FailLoud``. Future ``OnFailure(planner, max_replans=N)``
will allow adaptive recovery without infinite-replan risk.
"""
from __future__ import annotations

from typing import Any

from ballast.patterns.plan_execute._plan import Plan, PlannedStep


class FailLoud:
    """No re-planning. Step failure → ``PlanExecutionError`` raised by executor."""

    async def on_step_failure(
        self,
        plan: Plan,
        failed_step: PlannedStep,
        error: Exception,
        partial_outputs: dict[str, Any],
    ) -> Plan | None:
        return None


__all__ = ["FailLoud"]

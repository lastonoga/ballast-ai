"""Step + RePlanPolicy Protocols + StepContext vehicle.

Apps implement custom step kinds by writing a ``Step``-compatible class
and registering it under a name (see ``StepRegistry``). Apps implement
custom failure handling by writing a ``RePlanPolicy``-compatible class
and passing it to ``PlanAndExecute(replan_policy=...)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ballast.patterns.plan_execute._plan import Plan, PlannedStep


@dataclass
class StepContext:
    """Read-only context passed to ``Step.execute``."""

    plan: Plan
    """Full DAG being executed."""

    step: PlannedStep
    """The specific step being executed."""

    step_registry: Any
    """The ``StepRegistry`` — typed Any to avoid circular import; runtime ducktyped."""

    workflow_id: str | None = None
    """DBOS workflow id (None when running outside a workflow)."""


@runtime_checkable
class Step(Protocol):
    """How to execute one planned step.

    Instances are stateless; the framework calls ``execute()`` with the
    resolved inputs. Apps register a Step class per ``kind`` value the
    planner can emit; framework looks up by kind name.
    """

    async def execute(
        self,
        plan_input: Any,
        dep_outputs: dict[str, Any],
        ctx: StepContext,
    ) -> Any: ...


@runtime_checkable
class RePlanPolicy(Protocol):
    """When/whether to invoke planner again after a step failure.

    Returns:
      ``None`` — fail loud (raise ``PlanExecutionError`` with failed_step + partial_outputs).
      ``Plan`` — new DAG to continue with. Executor preserves completed-step outputs;
                 ``new_plan`` may reference them by step.id as dependencies.
    """

    async def on_step_failure(
        self,
        plan: Plan,
        failed_step: PlannedStep,
        error: Exception,
        partial_outputs: dict[str, Any],
    ) -> Plan | None: ...


__all__ = ["RePlanPolicy", "Step", "StepContext"]

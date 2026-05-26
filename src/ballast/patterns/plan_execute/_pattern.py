"""``PlanAndExecute`` — pattern entry point.

Same ``DBOSConfiguredInstance`` shape as ``MapReduce``. The unit stored
on ``self`` (planner + registry + replan policy) is never pickled per
step call because we live on a configured instance with a unique
``config_name``.
"""
from __future__ import annotations

import asyncio
import itertools
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dbos import DBOSConfiguredInstance

from ballast.durable import Durable
from ballast.patterns.plan_execute._errors import PlanExecutionError
from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._policies import FailLoud
from ballast.patterns.plan_execute._protocols import (
    RePlanPolicy, StepContext,
)
from ballast.patterns.plan_execute._registry import StepRegistry

if TYPE_CHECKING:
    from pydantic_ai import Agent


InT = TypeVar("InT")
OutT = TypeVar("OutT")

_instance_counter = itertools.count()


@Durable.dbos_class()
class PlanAndExecute(DBOSConfiguredInstance, Generic[InT, OutT]):
    """Plan-and-Execute pattern: planner emits DAG, framework executes nodes.

    Two-phase durable workflow:
      1. ``_plan_step`` — call planner.run(input) → ``Plan``.
      2. ``_execute_dag`` (orchestrator, not a step itself) — wave-by-wave
         traversal with ``asyncio.gather`` + semaphore; each step dispatch
         goes through ``_execute_step`` which IS a ``@Durable.step``.

    On replay, DBOS memoises completed steps; only the unfinished tail
    re-runs.
    """

    def __init__(
        self, *,
        planner: "Agent[None, Plan]",
        registry: StepRegistry,
        replan_policy: RePlanPolicy | None = None,
        max_parallel: int = 8,
    ) -> None:
        super().__init__(
            config_name=f"{type(self).__qualname__}-{next(_instance_counter)}",
        )
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._planner = planner
        self._registry = registry
        self._replan_policy: RePlanPolicy = replan_policy or FailLoud()
        self._max_parallel = max_parallel

    @Durable.workflow()
    async def run(self, input: InT) -> dict[str, Any]:
        """Returns ``{step_id: output}`` for every completed step."""
        plan = await self._plan_step(input)
        outputs = await self._execute_dag(input, plan)
        return outputs

    @Durable.step()
    async def _plan_step(self, input: InT) -> Plan:
        result = await self._planner.run(_serialize_for_planner(input))
        return result.output

    async def _execute_dag(self, plan_input: InT, plan: Plan) -> dict[str, Any]:
        """Wave-by-wave DAG traversal. Not a @Durable.step itself."""
        outputs: dict[str, Any] = {}
        pending: dict[str, PlannedStep] = {s.id: s for s in plan.steps}
        sem = asyncio.Semaphore(self._max_parallel)

        while pending:
            ready: list[PlannedStep] = [
                s for s in pending.values()
                if all(dep in outputs for dep in s.depends_on)
            ]
            if not ready:
                raise RuntimeError(
                    f"plan deadlock: {len(pending)} steps remain, none ready. "
                    f"Pending: {sorted(pending)}"
                )

            async def _run_one(step: PlannedStep) -> tuple[str, Any]:
                async with sem:
                    return step.id, await self._execute_step(
                        plan, step, plan_input, outputs,
                    )

            batch_results = await asyncio.gather(
                *(_run_one(s) for s in ready),
                return_exceptions=True,
            )

            new_plan: Plan | None = None
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    failed_step = ready[i]
                    new_plan = await self._replan_policy.on_step_failure(
                        plan=plan, failed_step=failed_step,
                        error=result, partial_outputs=outputs,
                    )
                    if new_plan is None:
                        raise PlanExecutionError(
                            f"step {failed_step.id!r} failed: {result}",
                            failed_step=failed_step,
                            partial_outputs=outputs,
                        ) from result
                    break  # rebuild pending from new_plan
                step_id, output = result  # type: ignore[misc]
                outputs[step_id] = output
                del pending[step_id]

            if new_plan is not None:
                plan = new_plan
                pending = {
                    s.id: s for s in plan.steps if s.id not in outputs
                }

        return outputs

    @Durable.step()
    async def _execute_step(
        self,
        plan: Plan,
        planned: PlannedStep,
        plan_input: InT,
        outputs: dict[str, Any],
    ) -> Any:
        """Execute one planned step — memoised per ``(config_name, args)``."""
        step_impl = self._registry.get_step(planned.kind)
        dep_outputs = {
            dep_id: outputs[dep_id] for dep_id in planned.depends_on
        }
        ctx = StepContext(
            plan=plan,
            step=planned,
            step_registry=self._registry,
            workflow_id=_current_workflow_id(),
        )
        return await step_impl.execute(plan_input, dep_outputs, ctx)


def _serialize_for_planner(input: Any) -> str:
    """Render the user-supplied plan input as a prompt string for the planner."""
    if isinstance(input, str):
        return input
    return repr(input)


def _current_workflow_id() -> str | None:
    """Best-effort fetch of current DBOS workflow id; None if unavailable."""
    try:
        return Durable.current_workflow_id()
    except Exception:  # noqa: BLE001
        return None


__all__ = ["PlanAndExecute"]

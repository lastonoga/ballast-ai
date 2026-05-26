"""Step + RePlanPolicy Protocols + StepContext vehicle."""
from __future__ import annotations

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import (
    RePlanPolicy, Step, StepContext,
)


def test_step_runtime_checkable() -> None:
    class _Stub:
        async def execute(self, plan_input, dep_outputs, ctx): return None
    assert isinstance(_Stub(), Step)

    class _Missing:
        pass
    assert not isinstance(_Missing(), Step)


def test_replan_policy_runtime_checkable() -> None:
    class _Stub:
        async def on_step_failure(self, plan, failed_step, error, partial_outputs):
            return None
    assert isinstance(_Stub(), RePlanPolicy)


def test_step_context_holds_plan_step_registry_workflow_id() -> None:
    plan = Plan(steps=[PlannedStep(id="a", kind="llm")])
    step = plan.steps[0]
    ctx = StepContext(plan=plan, step=step, step_registry=None, workflow_id="wf-1")
    assert ctx.plan is plan
    assert ctx.step is step
    assert ctx.step_registry is None
    assert ctx.workflow_id == "wf-1"


def test_step_context_workflow_id_optional() -> None:
    plan = Plan(steps=[PlannedStep(id="a", kind="llm")])
    ctx = StepContext(plan=plan, step=plan.steps[0], step_registry=None)
    assert ctx.workflow_id is None

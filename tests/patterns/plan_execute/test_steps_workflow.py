"""WorkflowStep — dispatches to a registered async workflow callable."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import WorkflowStep


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_workflow_step_invokes_callable_with_plan_input() -> None:
    captured = []

    async def my_wf(input):
        captured.append(input)
        return f"done-{input}"

    registry = StepRegistry()
    registry.register_workflow("my_wf", my_wf)
    step = PlannedStep(
        id="s1", kind="workflow", params={"workflow_name": "my_wf"},
    )
    out = await WorkflowStep(registry).execute(
        plan_input="payload", dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "done-payload"
    assert captured == ["payload"]


@pytest.mark.asyncio
async def test_workflow_step_uses_dep_output_when_input_from_set() -> None:
    captured = []

    async def my_wf(input):
        captured.append(input)
        return None

    registry = StepRegistry()
    registry.register_workflow("my_wf", my_wf)
    step = PlannedStep(
        id="s1", kind="workflow",
        params={"workflow_name": "my_wf", "input_from": "dep_a"},
    )
    await WorkflowStep(registry).execute(
        plan_input="ignored",
        dep_outputs={"dep_a": {"k": 1}},
        ctx=_ctx(step, registry),
    )
    assert captured == [{"k": 1}]


@pytest.mark.asyncio
async def test_workflow_step_unknown_workflow_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="workflow", params={"workflow_name": "missing"},
    )
    with pytest.raises(KeyError, match="missing"):
        await WorkflowStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )

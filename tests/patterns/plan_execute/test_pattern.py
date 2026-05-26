"""PlanAndExecute.run end-to-end."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ballast.patterns.plan_execute._errors import PlanExecutionError
from ballast.patterns.plan_execute._pattern import PlanAndExecute
from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._policies import FailLoud
from ballast.patterns.plan_execute._registry import StepRegistry


@dataclass
class _FakeRes:
    output: Any


class _FakePlanner:
    """Mimics pydantic-ai Agent[None, Plan].run."""
    def __init__(self, plan: Plan): self.plan = plan
    async def run(self, input: Any):
        return _FakeRes(self.plan)


@pytest.mark.asyncio
async def test_run_linear_plan_returns_dict_of_step_outputs(fresh_dbos_executor) -> None:
    plan = Plan(steps=[
        PlannedStep(id="a", kind="callable", params={"fn_name": "fn_a"}),
        PlannedStep(id="b", kind="callable", params={"fn_name": "fn_b"}, depends_on=["a"]),
    ])
    registry = StepRegistry.with_defaults()

    async def fn_a(*, plan_input, dep_outputs):
        return f"A({plan_input})"

    async def fn_b(*, plan_input, dep_outputs):
        return f"B({dep_outputs['a']})"

    registry.register_callable("fn_a", fn_a)
    registry.register_callable("fn_b", fn_b)

    pattern = PlanAndExecute(planner=_FakePlanner(plan), registry=registry)
    outputs = await pattern.run("INPUT")

    assert outputs == {"a": "A(INPUT)", "b": "B(A(INPUT))"}


@pytest.mark.asyncio
async def test_run_diamond_plan_executes_parallel_branches(fresh_dbos_executor) -> None:
    import asyncio

    plan = Plan(steps=[
        PlannedStep(id="root", kind="callable", params={"fn_name": "fn_root"}),
        PlannedStep(id="left", kind="callable", params={"fn_name": "fn_branch"}, depends_on=["root"]),
        PlannedStep(id="right", kind="callable", params={"fn_name": "fn_branch"}, depends_on=["root"]),
        PlannedStep(id="join", kind="callable", params={"fn_name": "fn_join"}, depends_on=["left", "right"]),
    ])
    registry = StepRegistry.with_defaults()

    async def fn_root(*, plan_input, dep_outputs): return "R"
    async def fn_branch(*, plan_input, dep_outputs):
        await asyncio.sleep(0.01)
        return f"B({dep_outputs['root']})"
    async def fn_join(*, plan_input, dep_outputs):
        return f"J({dep_outputs['left']}+{dep_outputs['right']})"

    registry.register_callable("fn_root",   fn_root)
    registry.register_callable("fn_branch", fn_branch)
    registry.register_callable("fn_join",   fn_join)

    pattern = PlanAndExecute(planner=_FakePlanner(plan), registry=registry)
    outputs = await pattern.run(None)

    assert outputs["root"] == "R"
    assert outputs["left"] == "B(R)"
    assert outputs["right"] == "B(R)"
    assert outputs["join"] == "J(B(R)+B(R))"


@pytest.mark.asyncio
async def test_run_empty_plan_returns_empty_dict(fresh_dbos_executor) -> None:
    pattern = PlanAndExecute(
        planner=_FakePlanner(Plan(steps=[])),
        registry=StepRegistry.with_defaults(),
    )
    out = await pattern.run("x")
    assert out == {}


@pytest.mark.asyncio
async def test_fail_loud_raises_plan_execution_error(fresh_dbos_executor) -> None:
    plan = Plan(steps=[
        PlannedStep(id="bad", kind="callable", params={"fn_name": "boom"}),
    ])
    registry = StepRegistry.with_defaults()

    async def boom(*, plan_input, dep_outputs):
        raise RuntimeError("kaboom")

    registry.register_callable("boom", boom)

    pattern = PlanAndExecute(
        planner=_FakePlanner(plan), registry=registry,
        replan_policy=FailLoud(),
    )
    with pytest.raises(PlanExecutionError) as exc:
        await pattern.run(None)

    assert exc.value.failed_step.id == "bad"
    assert exc.value.partial_outputs == {}
    assert "kaboom" in str(exc.value.__cause__)


@pytest.mark.asyncio
async def test_custom_replan_policy_continues_after_failure(fresh_dbos_executor) -> None:
    plan_v1 = Plan(steps=[
        PlannedStep(id="a", kind="callable", params={"fn_name": "ok"}),
        PlannedStep(id="b", kind="callable", params={"fn_name": "boom"}, depends_on=["a"]),
    ])
    plan_v2 = Plan(steps=[
        PlannedStep(id="a", kind="callable", params={"fn_name": "ok"}),
        PlannedStep(id="b_recovery", kind="callable", params={"fn_name": "ok"}, depends_on=["a"]),
    ])
    registry = StepRegistry.with_defaults()

    async def ok(*, plan_input, dep_outputs):
        return "OK"

    async def boom(*, plan_input, dep_outputs):
        raise RuntimeError("fail")

    registry.register_callable("ok", ok)
    registry.register_callable("boom", boom)

    class _SwapPlan:
        def __init__(self): self.calls = 0
        async def on_step_failure(self, plan, failed_step, error, partial_outputs):
            self.calls += 1
            return plan_v2 if self.calls == 1 else None

    pattern = PlanAndExecute(
        planner=_FakePlanner(plan_v1), registry=registry,
        replan_policy=_SwapPlan(),
    )
    outputs = await pattern.run(None)
    assert outputs == {"a": "OK", "b_recovery": "OK"}


@pytest.mark.asyncio
async def test_max_parallel_caps_concurrency(fresh_dbos_executor) -> None:
    import asyncio

    plan = Plan(steps=[
        PlannedStep(id=f"s{i}", kind="callable", params={"fn_name": "slow"})
        for i in range(5)
    ])
    registry = StepRegistry.with_defaults()

    in_flight = {"n": 0, "max": 0}

    async def slow(*, plan_input, dep_outputs):
        in_flight["n"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["n"])
        await asyncio.sleep(0.01)
        in_flight["n"] -= 1
        return None

    registry.register_callable("slow", slow)

    pattern = PlanAndExecute(
        planner=_FakePlanner(plan), registry=registry, max_parallel=2,
    )
    await pattern.run(None)
    assert in_flight["max"] <= 2

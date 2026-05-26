"""CallableStep — dispatches to a registered async function."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import CallableStep


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_callable_step_invokes_function_with_plan_input_and_dep_outputs() -> None:
    captured = {}

    async def my_fn(*, plan_input, dep_outputs, extra=None):
        captured["plan_input"] = plan_input
        captured["dep_outputs"] = dep_outputs
        captured["extra"] = extra
        return "result"

    registry = StepRegistry()
    registry.register_callable("my_fn", my_fn)
    step = PlannedStep(
        id="s1", kind="callable",
        params={"fn_name": "my_fn", "args": {"extra": "hello"}},
    )
    out = await CallableStep(registry).execute(
        plan_input={"x": 1}, dep_outputs={"a": "out_a"},
        ctx=_ctx(step, registry),
    )
    assert out == "result"
    assert captured["plan_input"] == {"x": 1}
    assert captured["dep_outputs"] == {"a": "out_a"}
    assert captured["extra"] == "hello"


@pytest.mark.asyncio
async def test_callable_step_args_optional() -> None:
    called = []

    async def my_fn(*, plan_input, dep_outputs):
        called.append((plan_input, dep_outputs))
        return None

    registry = StepRegistry()
    registry.register_callable("my_fn", my_fn)
    step = PlannedStep(id="s1", kind="callable", params={"fn_name": "my_fn"})
    await CallableStep(registry).execute(
        plan_input=42, dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert called == [(42, {})]


@pytest.mark.asyncio
async def test_callable_step_unknown_function_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="callable", params={"fn_name": "missing"},
    )
    with pytest.raises(KeyError, match="missing"):
        await CallableStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )

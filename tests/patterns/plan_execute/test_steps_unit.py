"""UnitStep — dispatches to a registered CoALAUnit via its 4-phase lifecycle."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase
from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import UnitStep


class _RecordingUnit(CoALABase[str, str, dict, str]):
    """Records each phase invocation order."""
    calls: list[str] = []

    async def observe(self, input):
        self.calls.append(f"observe({input})")
        return input.upper()

    async def retrieve(self, observation):
        self.calls.append(f"retrieve({observation})")
        return {"ctx": observation}

    async def act(self, observation, context):
        self.calls.append(f"act({observation},{context})")
        return f"acted-{observation}"

    async def learn(self, observation, context, output):
        self.calls.append(f"learn({output})")


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_unit_step_runs_four_phases_in_order_with_plan_input() -> None:
    unit = _RecordingUnit()
    unit.calls = []
    registry = StepRegistry()
    registry.register_unit("u", unit)
    step = PlannedStep(id="s1", kind="unit", params={"unit_name": "u"})
    out = await UnitStep(registry).execute(
        plan_input="hello", dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "acted-HELLO"
    assert unit.calls == [
        "observe(hello)",
        "retrieve(HELLO)",
        "act(HELLO,{'ctx': 'HELLO'})",
        "learn(acted-HELLO)",
    ]


@pytest.mark.asyncio
async def test_unit_step_uses_dep_output_when_input_from_set() -> None:
    unit = _RecordingUnit()
    unit.calls = []
    registry = StepRegistry()
    registry.register_unit("u", unit)
    step = PlannedStep(
        id="s1", kind="unit",
        params={"unit_name": "u", "input_from": "dep_a"},
    )
    await UnitStep(registry).execute(
        plan_input="ignored",
        dep_outputs={"dep_a": "from_dep"},
        ctx=_ctx(step, registry),
    )
    assert unit.calls[0] == "observe(from_dep)"


@pytest.mark.asyncio
async def test_unit_step_unknown_unit_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="unit", params={"unit_name": "missing"},
    )
    with pytest.raises(KeyError, match="missing"):
        await UnitStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )

"""``as_workflow`` adapter — wraps CoALAUnit as @Durable.workflow."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase, as_workflow


class _Recording(CoALABase[str, str, dict, str]):
    """Records each phase call for assertion."""
    calls: list[str] = []

    async def observe(self, input):
        self.calls.append(f"observe({input})")
        return input.upper()

    async def retrieve(self, observation):
        self.calls.append(f"retrieve({observation})")
        return {"ctx": observation + "_data"}

    async def act(self, observation, context):
        self.calls.append(f"act({observation}, {context})")
        return f"{observation}|{context['ctx']}"

    async def learn(self, observation, context, output):
        self.calls.append(f"learn(out={output})")


@pytest.mark.asyncio
async def test_workflow_runs_all_four_phases_in_order(
    fresh_dbos_executor: None,
) -> None:
    unit = _Recording()
    unit.calls = []
    runner = as_workflow(unit)
    out = await runner("hello")
    assert out == "HELLO|HELLO_data"
    assert unit.calls == [
        "observe(hello)",
        "retrieve(HELLO)",
        "act(HELLO, {'ctx': 'HELLO_data'})",
        "learn(out=HELLO|HELLO_data)",
    ]


@pytest.mark.asyncio
async def test_workflow_returns_act_output_not_learn(
    fresh_dbos_executor: None,
) -> None:
    class _Unit(CoALABase[str, str, dict, str]):
        async def retrieve(self, observation): return {}
        async def act(self, observation, context): return "from-act"
        async def learn(self, observation, context, output): return None

    runner = as_workflow(_Unit())
    out = await runner("x")
    assert out == "from-act"

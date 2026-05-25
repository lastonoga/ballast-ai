"""``as_tool`` adapter — wraps CoALAUnit as pydantic-ai Tool."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase, as_tool


class _Greeter(CoALABase[str, str, dict, str]):
    """Greets the user."""
    calls: list[str] = []

    async def observe(self, input):
        self.calls.append(f"observe({input})")
        return input

    async def retrieve(self, observation):
        self.calls.append(f"retrieve({observation})")
        return {"prefix": "Hi"}

    async def act(self, observation, context):
        self.calls.append(f"act({observation})")
        return f"{context['prefix']}, {observation}!"

    async def learn(self, observation, context, output):
        self.calls.append(f"learn({output})")


def test_tool_name_defaults_to_class_name() -> None:
    tool = as_tool(_Greeter())
    assert tool.name == "_Greeter"


def test_tool_description_defaults_to_class_docstring() -> None:
    tool = as_tool(_Greeter())
    assert (tool.description or "").strip() == "Greets the user."


def test_tool_overrides_take_precedence() -> None:
    tool = as_tool(_Greeter(), name="greet", description="custom desc")
    assert tool.name == "greet"
    assert tool.description == "custom desc"


@pytest.mark.asyncio
async def test_tool_invocation_runs_all_phases() -> None:
    unit = _Greeter()
    unit.calls = []
    tool = as_tool(unit)
    out = await tool.function(input="Alice")
    assert out == "Hi, Alice!"
    assert unit.calls == [
        "observe(Alice)",
        "retrieve(Alice)",
        "act(Alice)",
        "learn(Hi, Alice!)",
    ]

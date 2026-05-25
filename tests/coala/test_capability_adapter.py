"""``as_capability`` adapter — wraps CoALAUnit as pydantic-ai capability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import AgentRunResult, RunContext, RunUsage
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters

from ballast.coala import CoALABase, as_capability


# ---------------------------------------------------------------------------
# Helpers: minimal pydantic-ai stubs
# ---------------------------------------------------------------------------


def _make_request_context(text: str = "hello-msg") -> ModelRequestContext:
    """Build a minimal ModelRequestContext with a single user message."""
    rp = ModelRequestParameters(
        function_tools=[], output_tools=[], allow_text_output=True
    )
    msg = ModelRequest(parts=[UserPromptPart(content=text)])
    return ModelRequestContext(
        model=None,
        messages=[msg],
        model_settings=None,
        model_request_parameters=rp,
    )


def _make_ctx(deps: dict | None = None) -> RunContext[dict]:
    return RunContext(deps=deps or {}, model=None, usage=RunUsage())


def _make_result(output: str = "agent-output") -> "AgentRunResult[str]":
    return AgentRunResult(output=output)


# ---------------------------------------------------------------------------
# Test unit
# ---------------------------------------------------------------------------


class _Recording(CoALABase[ModelRequestContext, ModelRequestContext, dict, str]):
    """Records phase calls."""

    calls: list[str] = []

    async def observe(self, input: ModelRequestContext) -> ModelRequestContext:
        self.calls.append(f"observe({_text(input)})")
        return input

    async def retrieve(
        self, observation: ModelRequestContext
    ) -> dict:
        self.calls.append(f"retrieve({_text(observation)})")
        return {"ctx": "data"}

    async def act(self, observation: ModelRequestContext, context: dict) -> str:
        self.calls.append("act-NOT-CALLED-BY-FRAMEWORK")
        return "should-not-run"

    async def learn(
        self,
        observation: ModelRequestContext,
        context: dict,
        output: str,
    ) -> None:
        self.calls.append(f"learn({output})")


def _text(rc: ModelRequestContext) -> str:
    """Extract last user-prompt text from a ModelRequestContext."""
    for msg in reversed(rc.messages):
        for part in reversed(msg.parts):  # type: ignore[union-attr]
            if hasattr(part, "content") and isinstance(part.content, str):
                return part.content
    return repr(rc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capability_observe_and_retrieve_fire_before_request() -> None:
    unit = _Recording()
    unit.calls = []
    cap = as_capability(unit)
    ctx = _make_ctx()
    rc = _make_request_context("hello-msg")
    out = await cap.before_model_request(ctx, rc)
    assert out is rc
    assert unit.calls == ["observe(hello-msg)", "retrieve(hello-msg)"]


@pytest.mark.asyncio
async def test_capability_learn_fires_after_run() -> None:
    unit = _Recording()
    unit.calls = []
    cap = as_capability(unit)
    ctx = _make_ctx()
    await cap.before_model_request(ctx, _make_request_context("msg"))
    unit.calls = []
    result = _make_result("final-output")
    out = await cap.after_run(ctx, result=result)
    assert out is result
    assert unit.calls == ["learn(final-output)"]


@pytest.mark.asyncio
async def test_capability_gate_skips_learn() -> None:
    unit = _Recording()
    unit.calls = []
    cap = as_capability(unit, gate=lambda result: False)
    ctx = _make_ctx()
    await cap.before_model_request(ctx, _make_request_context("msg"))
    unit.calls = []
    out = await cap.after_run(ctx, result=_make_result())
    assert unit.calls == []


@pytest.mark.asyncio
async def test_capability_swallows_learn_exceptions() -> None:
    class _BrokenLearn(CoALABase[ModelRequestContext, ModelRequestContext, dict, str]):
        async def retrieve(self, observation: ModelRequestContext) -> dict:
            return {}

        async def act(
            self, observation: ModelRequestContext, context: dict
        ) -> str | None:
            return None

        async def learn(
            self,
            observation: ModelRequestContext,
            context: dict,
            output: str,
        ) -> None:
            raise RuntimeError("oops")

    cap = as_capability(_BrokenLearn())
    ctx = _make_ctx()
    await cap.before_model_request(ctx, _make_request_context("msg"))
    result = _make_result()
    out = await cap.after_run(ctx, result=result)
    assert out is result

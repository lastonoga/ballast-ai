"""LLMStep — agent invocation with templated prompt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import LLMStep


@dataclass
class _FakeResult:
    output: Any


class _RecordingAgent:
    """Mimics pydantic-ai Agent.run."""
    def __init__(self, output): self.output = output; self.prompts = []
    async def run(self, prompt):
        self.prompts.append(prompt)
        return _FakeResult(self.output)


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_llm_step_invokes_agent_with_rendered_prompt() -> None:
    registry = StepRegistry()
    agent = _RecordingAgent(output="summary")
    registry.register_agent("summarizer", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={"agent_name": "summarizer", "prompt_template": "Summarize: {plan_input}"},
    )
    out = await LLMStep(registry).execute(
        plan_input="raw text", dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "summary"
    assert agent.prompts == ["Summarize: raw text"]


@pytest.mark.asyncio
async def test_llm_step_substitutes_plan_input_attr() -> None:
    @dataclass
    class _Input:
        topic: str

    registry = StepRegistry()
    agent = _RecordingAgent(output="ok")
    registry.register_agent("ag", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={"agent_name": "ag", "prompt_template": "Topic={plan_input.topic}"},
    )
    await LLMStep(registry).execute(
        plan_input=_Input(topic="X"), dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert agent.prompts == ["Topic=X"]


@pytest.mark.asyncio
async def test_llm_step_substitutes_dep_output_whole_and_field() -> None:
    @dataclass
    class _D:
        title: str

    registry = StepRegistry()
    agent = _RecordingAgent(output="ok")
    registry.register_agent("ag", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={
            "agent_name": "ag",
            "prompt_template": "Whole={dep_a} Title={dep_b.title}",
        },
    )
    await LLMStep(registry).execute(
        plan_input=None,
        dep_outputs={"dep_a": "ALPHA", "dep_b": _D(title="BETA")},
        ctx=_ctx(step, registry),
    )
    assert agent.prompts == ["Whole=ALPHA Title=BETA"]


@pytest.mark.asyncio
async def test_llm_step_extracts_output_field_when_specified() -> None:
    @dataclass
    class _Result:
        summary: str
        debug: str

    registry = StepRegistry()
    agent = _RecordingAgent(output=_Result(summary="S", debug="D"))
    registry.register_agent("ag", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={
            "agent_name": "ag",
            "prompt_template": "x",
            "output_field": "summary",
        },
    )
    out = await LLMStep(registry).execute(
        plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "S"


@pytest.mark.asyncio
async def test_llm_step_unknown_agent_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="llm",
        params={"agent_name": "missing", "prompt_template": "x"},
    )
    with pytest.raises(KeyError, match="missing"):
        await LLMStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )

"""Unit tests for ``DivergentConvergent`` covering the
agent + projector contract (no ``diverge``/``synthesize`` methods on
agents — the pattern owns envelope→hypotheses mapping).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.patterns.divergent_convergent import (
    DivergentBranch,
    DivergentConvergent,
)


class _Idea(BaseModel):
    title: str


class _Ideas(BaseModel):
    """Envelope returned by mock divergent agents — mirrors the
    ``TodoIdeas { ideas: list[TodoIdea] }`` shape used by the notes-app."""
    ideas: list[_Idea]


@dataclass
class _AgentResult:
    """Structural stand-in for pydantic-ai's ``AgentRunResult``."""
    output: object


class _MockDivergentAgent:
    """Returns a fixed envelope per call. The pattern projects via
    the ``hypotheses`` callable supplied to ``DivergentConvergent``."""

    def __init__(self, ideas: list[_Idea]) -> None:
        self._ideas = ideas
        self.calls = 0

    async def run(self, task: str) -> _AgentResult:
        del task
        self.calls += 1
        return _AgentResult(output=_Ideas(ideas=list(self._ideas)))


class _MockSynthesizer:
    """Returns the first idea unchanged. Receives a string prompt
    (built by the pattern via ``format_synth_prompt``)."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def run(self, prompt: str) -> _AgentResult:
        self.last_prompt = prompt
        # Pretend we picked the first candidate from the rendered prompt.
        return _AgentResult(output=_Idea(title="winner"))


@pytest.mark.asyncio
async def test_pattern_invokes_hypotheses_projector_per_branch() -> None:
    """``DivergentConvergent`` should call ``hypotheses(env)`` exactly
    once per successful branch and feed the merged pool into the
    synthesizer via ``format_synth_prompt``."""
    agent_a = _MockDivergentAgent([_Idea(title="a1"), _Idea(title="a2")])
    agent_b = _MockDivergentAgent([_Idea(title="b1")])
    synth = _MockSynthesizer()

    projector_calls: list[_Ideas] = []

    def hypotheses(env: _Ideas) -> list[_Idea]:
        projector_calls.append(env)
        return env.ideas

    prompt_calls: list[tuple[str, list[_Idea]]] = []

    def format_synth_prompt(task: str, candidates: list[_Idea]) -> str:
        prompt_calls.append((task, list(candidates)))
        return f"task={task};n={len(candidates)}"

    dc = DivergentConvergent[str, _Ideas, _Idea, _Idea](
        branches=(
            DivergentBranch(label="a", agent=agent_a),
            DivergentBranch(label="b", agent=agent_b),
        ),
        synthesizer=synth,
        hypotheses=hypotheses,
        format_synth_prompt=format_synth_prompt,
        min_hypotheses=2,
        divergent_concurrency=2,
        config_name=f"test-dc-{uuid4()}",
    )

    result = await dc.run("topic")

    assert result == _Idea(title="winner")
    assert agent_a.calls == 1
    assert agent_b.calls == 1
    assert len(projector_calls) == 2
    assert len(prompt_calls) == 1
    task, candidates = prompt_calls[0]
    assert task == "topic"
    # Merged pool: agent_a's two ideas + agent_b's one idea, in order.
    assert [c.title for c in candidates] == ["a1", "a2", "b1"]
    assert synth.last_prompt == "task=topic;n=3"

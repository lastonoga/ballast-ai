"""DivergentConvergent on_progress callback parameter."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.patterns.divergent_convergent import (
    DivergentBranch,
    DivergentConvergent,
)
from ballast.patterns.divergent_convergent.events import (
    BranchCompleted,
    BranchEnqueued,
    ConvergeCompleted,
    ConvergeStarted,
    DivergentEvent,
)


# ── Minimal fixtures (mirrored from test_divergent_convergent.py) ─────────────


class _Idea(BaseModel):
    title: str


class _Ideas(BaseModel):
    ideas: list[_Idea]


@dataclass
class _AgentResult:
    output: object


class _MockDivergentAgent:
    def __init__(self, ideas: list[_Idea]) -> None:
        self._ideas = ideas

    async def run(self, task: str) -> _AgentResult:
        del task
        return _AgentResult(output=_Ideas(ideas=list(self._ideas)))


class _MockSynthesizer:
    async def run(self, prompt: str) -> _AgentResult:
        del prompt
        return _AgentResult(output=_Idea(title="winner"))


def _make_dc(*, on_progress=None) -> DivergentConvergent:
    """Minimal valid DivergentConvergent instance (two branches, no dedup/verifier)."""
    return DivergentConvergent[str, _Ideas, _Idea, _Idea](
        branches=(
            DivergentBranch(label="a", agent=_MockDivergentAgent([_Idea(title="a1"), _Idea(title="a2")])),
            DivergentBranch(label="b", agent=_MockDivergentAgent([_Idea(title="b1")])),
        ),
        synthesizer=_MockSynthesizer(),
        hypotheses=lambda env: env.ideas,
        min_hypotheses=2,
        divergent_concurrency=2,
        config_name=f"test-dc-progress-{uuid4()}",
        on_progress=on_progress,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_progress_callback_receives_all_event_types(
    fresh_dbos_executor: None,
) -> None:
    """When on_progress callback is provided, it receives every typed event."""
    received: list[DivergentEvent] = []

    async def my_callback(event: DivergentEvent) -> None:
        received.append(event)

    dc = _make_dc(on_progress=my_callback)
    result = await dc.run("topic")

    assert result == _Idea(title="winner")
    assert len(received) > 0

    event_type_names = {type(e).__name__ for e in received}
    assert "BranchEnqueued" in event_type_names
    assert "BranchCompleted" in event_type_names
    assert "ConvergeStarted" in event_type_names
    assert "ConvergeCompleted" in event_type_names

    # Verify ordering: at least one BranchEnqueued before any BranchCompleted
    enqueued_idx = next(i for i, e in enumerate(received) if isinstance(e, BranchEnqueued))
    completed_idx = next(i for i, e in enumerate(received) if isinstance(e, BranchCompleted))
    assert enqueued_idx < completed_idx

    # ConvergeStarted before ConvergeCompleted
    start_idx = next(i for i, e in enumerate(received) if isinstance(e, ConvergeStarted))
    done_idx = next(i for i, e in enumerate(received) if isinstance(e, ConvergeCompleted))
    assert start_idx < done_idx


@pytest.mark.asyncio
async def test_on_progress_callback_exceptions_are_swallowed(
    fresh_dbos_executor: None,
) -> None:
    """A throwing callback must not break the pattern run."""

    async def bad_callback(event: DivergentEvent) -> None:
        raise RuntimeError("callback boom")

    dc = _make_dc(on_progress=bad_callback)
    # Run should still complete normally despite the callback always raising.
    result = await dc.run("topic")
    assert result == _Idea(title="winner")


@pytest.mark.asyncio
async def test_on_progress_callback_optional_default_none(
    fresh_dbos_executor: None,
) -> None:
    """Pattern works without on_progress (backward compatibility)."""
    dc = _make_dc(on_progress=None)
    result = await dc.run("topic")
    assert result == _Idea(title="winner")

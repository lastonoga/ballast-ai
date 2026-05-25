"""DriftEngine.maybe_check — orchestration."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from ballast.drift._core import DriftEngine
from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import DriftCheckSignal, DriftContext
from ballast.drift._verdict import DefaultDriftVerdict


# ---- Fakes -----------------------------------------------------------------

@dataclass
class _NeverFires:
    def should_check(self, _sig): return False

@dataclass
class _AlwaysFires:
    def should_check(self, _sig): return True


class _FixedWindow:
    def __init__(self, msgs): self.msgs = msgs
    async def slice(self, ctx): return list(self.msgs)


class _FixedGoal:
    def __init__(self, text): self.text = text
    async def goal(self, ctx): return self.text


class _FixedPrompt:
    def build(self, goal, trace): return f"goal={goal}|n={len(trace)}"


class _FakeJudge:
    """Mimics pydantic-ai Agent.run for typed output."""
    def __init__(self, *, verdict=None, raises=None):
        self.verdict = verdict
        self.raises = raises
        self.calls = 0

    async def run(self, prompt, *, output_type):
        self.calls += 1
        if self.raises:
            raise self.raises
        return _FakeJudgeResult(self.verdict)


@dataclass
class _FakeJudgeResult:
    output: Any


class _RecordingHandler:
    def __init__(self, *, raises=None):
        self.calls = []
        self.raises = raises
    async def handle(self, verdict, ctx):
        self.calls.append(verdict)
        if self.raises:
            raise self.raises


def _sig() -> DriftCheckSignal:
    return DriftCheckSignal(step_index=1, tool_calls=0, tokens_used=0, seconds_elapsed=0.0)


def _ctx(msgs=()) -> DriftContext:
    return DriftContext(messages=list(msgs), run_ctx=None, workflow_input=None)


# ---- Tests -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_none_when_strategy_skips() -> None:
    judge = _FakeJudge()
    engine = DriftEngine(
        strategy=_NeverFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is None
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_returns_none_on_empty_trace() -> None:
    judge = _FakeJudge()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[],
    )
    out = await engine.maybe_check(_sig(), _ctx())
    assert out is None
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_judge_exception_swallowed_returns_none(caplog) -> None:
    import logging
    caplog.set_level(logging.ERROR, logger="ballast.drift")
    judge = _FakeJudge(raises=RuntimeError("model down"))
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is None
    assert "judge failed" in caplog.text.lower() or "model down" in caplog.text


@pytest.mark.asyncio
async def test_should_not_interrupt_skips_handlers() -> None:
    v = DefaultDriftVerdict(should_interrupt=False, reason="ok", score=1.0, category="on_track")
    judge = _FakeJudge(verdict=v)
    handler = _RecordingHandler()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[handler],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is v
    assert handler.calls == []


@pytest.mark.asyncio
async def test_should_interrupt_fires_handlers_in_order() -> None:
    v = DefaultDriftVerdict(should_interrupt=True, reason="drifted", score=0.1, category="drifted")
    judge = _FakeJudge(verdict=v)
    h1, h2 = _RecordingHandler(), _RecordingHandler()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[h1, h2],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is v
    assert h1.calls == [v]
    assert h2.calls == [v]


@pytest.mark.asyncio
async def test_one_handler_failure_does_not_block_others(caplog) -> None:
    import logging
    caplog.set_level(logging.ERROR, logger="ballast.drift")
    v = DefaultDriftVerdict(should_interrupt=True, reason="x", score=0.1, category="drifted")
    judge = _FakeJudge(verdict=v)
    h_bad = _RecordingHandler(raises=RuntimeError("handler boom"))
    h_good = _RecordingHandler()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[h_bad, h_good],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is v
    assert h_bad.calls == [v]      # attempted
    assert h_good.calls == [v]     # ran after the failure
    assert "handler" in caplog.text.lower() or "boom" in caplog.text


@pytest.mark.asyncio
async def test_goal_drift_error_from_handler_propagates() -> None:
    v = DefaultDriftVerdict(should_interrupt=True, reason="hard", score=0.0, category="drifted")
    judge = _FakeJudge(verdict=v)
    h_raise = _RecordingHandler(raises=GoalDriftError(v))
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[h_raise],
    )
    with pytest.raises(GoalDriftError):
        await engine.maybe_check(_sig(), _ctx([1]))

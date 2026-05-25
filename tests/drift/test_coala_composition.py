"""goal_drift_as_unit — wrap DriftEngine as a CoALAUnit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ballast.coala import CoALAUnit
from ballast.drift._coala import goal_drift_as_unit
from ballast.drift._core import DriftEngine
from ballast.drift._protocols import DriftContext
from ballast.drift._verdict import DefaultDriftVerdict


class _AlwaysFires:
    def should_check(self, _sig): return True


class _Window:
    async def slice(self, ctx): return [1]


class _Goal:
    async def goal(self, ctx): return "g"


class _Prompt:
    def build(self, goal, trace): return "p"


class _Judge:
    def __init__(self, v): self.v = v
    async def run(self, p, *, output_type):
        return _R(self.v)


@dataclass
class _R:
    output: Any


class _Recording:
    def __init__(self): self.calls = []
    async def handle(self, v, ctx): self.calls.append(v)


@pytest.mark.asyncio
async def test_goal_drift_as_unit_satisfies_coala_unit_protocol() -> None:
    v = DefaultDriftVerdict(should_interrupt=False, reason="ok", score=1.0, category="on_track")
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_Window(), goal_source=_Goal(),
        prompt=_Prompt(), judge=_Judge(v), handlers=[],
    )
    unit = goal_drift_as_unit(engine)
    assert isinstance(unit, CoALAUnit)


@pytest.mark.asyncio
async def test_unit_calls_engine_in_retrieve() -> None:
    v = DefaultDriftVerdict(should_interrupt=True, reason="d", score=0.0, category="drifted")
    handler = _Recording()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_Window(), goal_source=_Goal(),
        prompt=_Prompt(), judge=_Judge(v), handlers=[handler],
    )
    unit = goal_drift_as_unit(engine)

    ctx_in = DriftContext(messages=[1], run_ctx=None, workflow_input=None)
    obs = await unit.observe(ctx_in)
    verdict = await unit.retrieve(obs)
    assert verdict is v

    out = await unit.act(obs, verdict)
    # act fires handlers; with should_interrupt=True they ran during retrieve
    # (since retrieve calls engine.maybe_check, handlers ran there). Verify.
    assert handler.calls == [v]
    assert out is v

    await unit.learn(obs, verdict, out)  # no-op

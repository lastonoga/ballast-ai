"""Built-in DriftHandler implementations + GoalDriftError."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pytest

from ballast.drift._handlers import (
    Compose, EmitDriftEvent, EscalateToHITL, GoalDriftError,
    LogOnly, RaiseDriftError,
)
from ballast.drift._protocols import DriftContext
from ballast.drift._verdict import DefaultDriftVerdict


def _verdict(should_interrupt=True, reason="r", score=0.2, cat="drifted"):
    return DefaultDriftVerdict(
        should_interrupt=should_interrupt, reason=reason,
        score=score, category=cat,
    )


def _ctx() -> DriftContext:
    return DriftContext(messages=[], run_ctx=None, workflow_input=None)


@pytest.mark.asyncio
async def test_log_only_writes_warning(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="ballast.drift")
    await LogOnly().handle(_verdict(reason="off-topic"), _ctx())
    assert "off-topic" in caplog.text


@pytest.mark.asyncio
async def test_raise_drift_error_raises_goal_drift_error() -> None:
    with pytest.raises(GoalDriftError) as exc:
        await RaiseDriftError().handle(_verdict(reason="hard fail"), _ctx())
    assert exc.value.verdict.reason == "hard fail"


@pytest.mark.asyncio
async def test_compose_runs_in_order_and_isolates_failures() -> None:
    calls: list[str] = []

    class _Ok:
        def __init__(self, tag): self.tag = tag
        async def handle(self, v, ctx):
            calls.append(self.tag)

    class _Bad:
        async def handle(self, v, ctx):
            calls.append("bad-attempted")
            raise RuntimeError("boom")

    await Compose(_Ok("a"), _Bad(), _Ok("b")).handle(_verdict(), _ctx())
    assert calls == ["a", "bad-attempted", "b"]


@pytest.mark.asyncio
async def test_compose_propagates_goal_drift_error() -> None:
    # RaiseDriftError's GoalDriftError MUST propagate through Compose,
    # so callers can wire [LogOnly, RaiseDriftError] and still get hard-fail.
    with pytest.raises(GoalDriftError):
        await Compose(LogOnly(), RaiseDriftError()).handle(_verdict(), _ctx())


@pytest.mark.asyncio
async def test_emit_drift_event_calls_provided_sink() -> None:
    seen: list[dict[str, Any]] = []

    async def sink(event_name: str, payload: dict) -> None:
        seen.append({"name": event_name, "payload": payload})

    h = EmitDriftEvent(sink=sink, event_name="goal_drift")
    v = _verdict(reason="off topic")
    await h.handle(v, _ctx())
    assert len(seen) == 1
    assert seen[0]["name"] == "goal_drift"
    assert seen[0]["payload"]["reason"] == "off topic"


@pytest.mark.asyncio
async def test_escalate_to_hitl_calls_channel_request_blocking() -> None:
    requested = []

    class _Card:
        def __init__(self, verdict): self.verdict = verdict

    class _FakeChannel:
        async def request(self, payload, *, timeout=None):
            requested.append({"payload": payload, "timeout": timeout})
            return None  # verdict shape — not used here

    h = EscalateToHITL(
        channel=_FakeChannel(),  # type: ignore[arg-type]
        card_factory=_Card,
        timeout=timedelta(minutes=5),
    )
    await h.handle(_verdict(reason="drift"), _ctx())
    assert len(requested) == 1
    assert isinstance(requested[0]["payload"], _Card)
    assert requested[0]["timeout"] == timedelta(minutes=5)

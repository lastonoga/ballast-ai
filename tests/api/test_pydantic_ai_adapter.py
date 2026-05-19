"""Tests for `make_runner` — the pydantic-ai → AgentRunner adapter (F2+F5)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.api.streaming import StreamEvent, make_runner
from pydantic_ai_stateflow.api.streaming.kinds import StreamEventKind
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody


class ChatReply(BaseModel):
    reply: str = ""


@dataclass
class _FakeResult:
    snapshots: list[Any]

    async def stream_output(self, *, debounce_by: float = 0.05) -> AsyncIterator[Any]:
        del debounce_by
        for s in self.snapshots:
            yield s


class _FakeAgent:
    """Mimics enough of `pydantic_ai.Agent` for `make_runner` to drive it."""

    def __init__(
        self,
        snapshots: list[Any] | None = None,
        *,
        raise_during_stream: Exception | None = None,
    ) -> None:
        self.snapshots = snapshots or []
        self.last_prompt: str | None = None
        self.last_deps: Any = None
        self._raise = raise_during_stream

    def run_stream(
        self, prompt: str, *, deps: Any = None,
    ) -> _CM:  # noqa: D401
        self.last_prompt = prompt
        self.last_deps = deps
        if self._raise is not None:
            raise self._raise
        return _CM(_FakeResult(self.snapshots))


class _CM:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result

    async def __aenter__(self) -> _FakeResult:
        return self._result

    async def __aexit__(self, *exc: object) -> None:
        return None


async def _collect(runner: Any) -> list[StreamEvent]:
    out: list[StreamEvent] = []
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
    )
    async for ev in runner(
        thread_id=uuid4(), run_id=uuid4(), message=body, tenant_id=uuid4(),
    ):
        out.append(ev)
    return out


def _kinds(events: list[StreamEvent]) -> list[str]:
    return [e.kind for e in events]


def _deltas(events: list[StreamEvent]) -> list[str]:
    return [
        e.data["delta"]
        for e in events
        if e.kind == StreamEventKind.TEXT_MESSAGE_CONTENT.value
    ]


@pytest.mark.asyncio
async def test_make_runner_emits_canonical_event_sequence() -> None:
    agent = _FakeAgent(
        snapshots=[ChatReply(reply="He"), ChatReply(reply="Hello"), ChatReply(reply="Hello!")],
    )
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    events = await _collect(runner)
    assert _kinds(events) == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    # message_id is consistent across start → content × N → end
    start = events[1]
    contents = events[2:5]
    end = events[5]
    mid = start.data["messageId"]
    assert all(c.data["messageId"] == mid for c in contents)
    assert end.data["messageId"] == mid


@pytest.mark.asyncio
async def test_make_runner_diffs_against_last_emitted_text() -> None:
    agent = _FakeAgent(
        snapshots=[ChatReply(reply="He"), ChatReply(reply="Hello"), ChatReply(reply="Hello!")],
    )
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    events = await _collect(runner)
    assert _deltas(events) == ["He", "llo", "!"]


@pytest.mark.asyncio
async def test_make_runner_handles_revising_snapshot_suffix() -> None:
    # First "Hello" then revised to "Hello, " — suffix delta.
    agent = _FakeAgent(
        snapshots=[ChatReply(reply="Hello"), ChatReply(reply="Hello, ")],
    )
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    events = await _collect(runner)
    assert _deltas(events) == ["Hello", ", "]


@pytest.mark.asyncio
async def test_make_runner_handles_shorter_snapshot_full_reemit() -> None:
    # Pathological: "Hello!" then partial-validation shortens to "Hi".
    # Adapter must re-emit "Hi" as full text (never a negative diff).
    agent = _FakeAgent(
        snapshots=[ChatReply(reply="Hello!"), ChatReply(reply="Hi")],
    )
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    events = await _collect(runner)
    assert _deltas(events) == ["Hello!", "Hi"]


@pytest.mark.asyncio
async def test_make_runner_callable_text_field() -> None:
    class Nested(BaseModel):
        class Inner(BaseModel):
            bar: str = ""

        foo: Nested.Inner = None  # type: ignore[assignment]

    Nested.model_rebuild()

    snaps = [
        Nested(foo=Nested.Inner(bar="ab")),
        Nested(foo=Nested.Inner(bar="abcd")),
    ]
    agent = _FakeAgent(snapshots=snaps)
    runner = make_runner(
        agent,  # type: ignore[arg-type]
        text_field=lambda out: out.foo.bar,
    )
    events = await _collect(runner)
    assert _deltas(events) == ["ab", "cd"]


@pytest.mark.asyncio
async def test_make_runner_skips_empty_snapshots() -> None:
    agent = _FakeAgent(
        snapshots=[ChatReply(reply=""), ChatReply(reply="hi")],
    )
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    events = await _collect(runner)
    assert _deltas(events) == ["hi"]


class _RaisingAgent:
    def run_stream(self, prompt: str, *, deps: Any = None) -> _RaisingCM:
        del prompt, deps
        return _RaisingCM()


class _RaisingCM:
    async def __aenter__(self) -> _RaisingResult:
        return _RaisingResult()

    async def __aexit__(self, *exc: object) -> None:
        return None


class _RaisingResult:
    async def stream_output(self, *, debounce_by: float = 0.05) -> AsyncIterator[Any]:
        del debounce_by
        # need at least one yield before raising so the async generator
        # protocol kicks in; raise on next iteration
        yield ChatReply(reply="ok")
        raise RuntimeError("boom")


async def _drain(runner: Any, sink: list[StreamEvent]) -> None:
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
    )
    async for ev in runner(
        thread_id=uuid4(), run_id=uuid4(), message=body, tenant_id=uuid4(),
    ):
        sink.append(ev)


@pytest.mark.asyncio
async def test_make_runner_emits_run_error_on_exception() -> None:
    runner = make_runner(_RaisingAgent(), text_field="reply")  # type: ignore[arg-type]
    events: list[StreamEvent] = []
    with pytest.raises(RuntimeError, match="boom"):
        await _drain(runner, events)
    error_events = [e for e in events if e.kind == "RUN_ERROR"]
    assert len(error_events) == 1
    assert error_events[0].data["message"] == "boom"


@pytest.mark.asyncio
async def test_make_runner_passes_prompt_extracted_from_parts() -> None:
    agent = _FakeAgent(snapshots=[ChatReply(reply="ok")])
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    body = _PostMessageBody.model_validate(
        {
            "role": "user",
            "parts": [
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "world!"},
            ],
        },
    )
    async for _ev in runner(
        thread_id=uuid4(), run_id=uuid4(), message=body, tenant_id=uuid4(),
    ):
        pass
    assert agent.last_prompt == "Hello, world!"


@pytest.mark.asyncio
async def test_make_runner_forwards_deps() -> None:
    agent = _FakeAgent(snapshots=[ChatReply(reply="ok")])
    sentinel = object()
    runner = make_runner(agent, text_field="reply", deps=sentinel)  # type: ignore[arg-type]
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
    )
    async for _ev in runner(
        thread_id=uuid4(), run_id=uuid4(), message=body, tenant_id=uuid4(),
    ):
        pass
    assert agent.last_deps is sentinel


@pytest.mark.asyncio
async def test_make_runner_correlates_thread_and_run_id() -> None:
    agent = _FakeAgent(snapshots=[ChatReply(reply="ok")])
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
    )
    tid, rid = uuid4(), uuid4()
    events: list[StreamEvent] = []
    async for ev in runner(
        thread_id=tid, run_id=rid, message=body, tenant_id=uuid4(),
    ):
        events.append(ev)
    started = next(e for e in events if e.kind == "RUN_STARTED")
    finished = next(e for e in events if e.kind == "RUN_FINISHED")
    assert started.data == {"threadId": str(tid), "runId": str(rid)}
    assert finished.data == {"threadId": str(tid), "runId": str(rid)}

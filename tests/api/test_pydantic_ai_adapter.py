"""Tests for `make_runner` — the pydantic-ai → AgentRunner adapter.

These cover the TEXT-only path (deps forwarding, prompt extraction,
text-delta diffing, run-error propagation). The TOOL-CALL path lives in
`test_pydantic_ai_adapter_tool_calls.py` so this file stays focused on
diffing semantics and the FakeAgent-based fast tests.

The fake `_FakeAgent` mimics enough of `agent.iter` (a single
`ModelRequestNode` that streams a series of `TextPart` snapshots) for the
adapter to drive it. We deliberately don't import the real
`pydantic_ai.Agent` class methods here — the helpers
`Agent.is_model_request_node` / `is_call_tools_node` are duck-typed by
the adapter and the fake nodes carry the right markers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)

from pydantic_ai_stateflow.api.streaming import StreamEvent, make_runner
from pydantic_ai_stateflow.api.streaming.kinds import StreamEventKind
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody


class ChatReply(BaseModel):
    reply: str = ""


@dataclass
class _FakeRunResult:
    output: Any


@dataclass
class _FakeCtx:
    pass


@dataclass
class _FakeModelStream:
    """Async-context-manager streaming a list of pre-baked events."""

    events: list[Any]

    async def __aenter__(self) -> _FakeModelStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for ev in self.events:
            yield ev


@dataclass
class _FakeModelRequestNode:
    """Stand-in for pydantic-ai's ``ModelRequestNode``.

    The adapter's ``Agent.is_model_request_node(node)`` check uses
    pydantic-ai's classmethod; we patch the helpers in the fake-iter
    fixture below.
    """

    events: list[Any]

    def stream(self, _ctx: _FakeCtx) -> _FakeModelStream:
        return _FakeModelStream(self.events)


@dataclass
class _FakeAgentRun:
    nodes: list[Any]
    ctx: _FakeCtx = field(default_factory=_FakeCtx)
    result: _FakeRunResult | None = None

    async def __aenter__(self) -> _FakeAgentRun:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for n in self.nodes:
            yield n


class _FakeAgent:
    """Mimics enough of `pydantic_ai.Agent` for `make_runner` to drive it."""

    def __init__(
        self,
        snapshots: list[ChatReply] | None = None,
        *,
        raise_during_stream: Exception | None = None,
    ) -> None:
        self.snapshots = snapshots or []
        self.last_prompt: str | None = None
        self.last_deps: Any = None
        self._raise = raise_during_stream

    def iter(self, prompt: str, *, deps: Any = None) -> _FakeAgentRun:
        self.last_prompt = prompt
        self.last_deps = deps
        if self._raise is not None:
            # Raise *during* iteration so the adapter's try/except can
            # catch and convert to RUN_ERROR.
            events: list[Any] = [_RaisingEvent(self._raise)]
        else:
            events = self._events_from_snapshots()
        node = _FakeModelRequestNode(events)
        final = self.snapshots[-1] if self.snapshots else None
        return _FakeAgentRun(
            nodes=[node], result=_FakeRunResult(output=final),
        )

    def _events_from_snapshots(self) -> list[Any]:
        """Translate text-snapshot diffs into the pydantic-ai event stream
        the adapter expects: one PartStartEvent(TextPart) for the first
        non-empty snapshot, then PartDeltaEvent(TextPartDelta) for each
        subsequent change.

        For prefix-revising snapshots (Hello! → Hi), we emit a new
        PartStartEvent so the adapter's "fall back to full re-emit" branch
        is exercised end-to-end.
        """
        events: list[Any] = []
        last = ""
        for snap in self.snapshots:
            current = snap.reply or ""
            if not current or current == last:
                continue
            if not last:
                events.append(
                    PartStartEvent(index=0, part=TextPart(content=current)),
                )
            elif current.startswith(last):
                events.append(
                    PartDeltaEvent(
                        index=0,
                        delta=TextPartDelta(content_delta=current[len(last):]),
                    ),
                )
            else:
                # Prefix revision — model rare path. Emit a *new* part-start
                # at the same index. The adapter's _diff_text will fall
                # back to a full re-emit (no negative diff).
                events.append(
                    PartStartEvent(index=0, part=TextPart(content=current)),
                )
            last = current
        return events


@dataclass
class _RaisingEvent:
    """Sentinel event the fake model-stream raises when yielded."""

    exc: BaseException


# Patch _FakeModelStream.__aiter__ to honor _RaisingEvent sentinels.
_orig_aiter = _FakeModelStream.__aiter__


async def _aiter_with_raise(self: _FakeModelStream) -> AsyncIterator[Any]:
    for ev in self.events:
        if isinstance(ev, _RaisingEvent):
            raise ev.exc
        yield ev


_FakeModelStream.__aiter__ = _aiter_with_raise  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _patch_agent_node_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Agent.is_model_request_node`` / ``is_call_tools_node`` /
    ``is_end_node`` recognize our fake nodes.

    The adapter uses these classmethods to discriminate graph nodes; with
    fake nodes they'd all return False and the runner would emit nothing.
    """
    from pydantic_ai import Agent

    def _is_mr(node: Any) -> bool:
        return isinstance(node, _FakeModelRequestNode)

    def _is_ct(_node: Any) -> bool:
        return False

    def _is_end(_node: Any) -> bool:
        return False

    monkeypatch.setattr(Agent, "is_model_request_node", staticmethod(_is_mr))
    monkeypatch.setattr(Agent, "is_call_tools_node", staticmethod(_is_ct))
    monkeypatch.setattr(Agent, "is_end_node", staticmethod(_is_end))


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
    """The `text_field=lambda out: ...` extractor receives the final
    `agent_run.result.output`; we emit the suffix the part-stream missed.
    Here the snapshot stream is empty (no text parts in the model frames),
    so the entire callable-extracted text is emitted as one delta.
    """

    class Inner(BaseModel):
        bar: str = ""

    class Nested(BaseModel):
        foo: Inner = Inner()

    @dataclass
    class _Custom(_FakeAgent):
        # Just emit a single TextPart so the diffing path is exercised.
        pass

    agent = _FakeAgent(snapshots=[])
    # Manually inject the final output (the snapshot-empty path used to
    # have a custom Nested-snapshot streaming; we now rely on the final
    # output emit).
    agent.snapshots = []
    nested = Nested(foo=Inner(bar="abcd"))

    def _iter(prompt: str, *, deps: Any = None) -> _FakeAgentRun:
        agent.last_prompt = prompt
        agent.last_deps = deps
        return _FakeAgentRun(
            nodes=[_FakeModelRequestNode([])],
            result=_FakeRunResult(output=nested),
        )

    agent.iter = _iter  # type: ignore[method-assign]
    runner = make_runner(
        agent,  # type: ignore[arg-type]
        text_field=lambda out: out.foo.bar,
    )
    events = await _collect(runner)
    assert _deltas(events) == ["abcd"]


@pytest.mark.asyncio
async def test_make_runner_skips_empty_snapshots() -> None:
    agent = _FakeAgent(
        snapshots=[ChatReply(reply=""), ChatReply(reply="hi")],
    )
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
    events = await _collect(runner)
    assert _deltas(events) == ["hi"]


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
    agent = _FakeAgent(raise_during_stream=RuntimeError("boom"))
    runner = make_runner(agent, text_field="reply")  # type: ignore[arg-type]
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
async def test_make_runner_forwards_static_deps() -> None:
    """When ``deps`` is a non-callable value it's passed through unchanged."""
    agent = _FakeAgent(snapshots=[ChatReply(reply="ok")])

    @dataclass(frozen=True)
    class _Sentinel:
        # Use a frozen dataclass — NOT callable, so the adapter takes
        # the "static value" branch rather than the factory branch.
        token: str = "x"

    sentinel = _Sentinel()
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

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.streaming.ag_ui import AGUIEncoder
from pydantic_ai_stateflow.api.streaming.kinds import StreamEventKind
from pydantic_ai_stateflow.api.streaming.router import (
    StreamEvent,
    build_streaming_router,
)
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


def _parse(frame: bytes) -> tuple[str, dict[str, object]]:
    text = frame.decode("utf-8")
    assert text.endswith("\n\n")
    head, payload = text.rstrip("\n").split("\ndata: ", 1)
    assert head.startswith("event: ")
    return head[len("event: "):], json.loads(payload)


def test_stream_event_run_started_constructor() -> None:
    tid, rid = uuid4(), uuid4()
    ev = StreamEvent.run_started(thread_id=tid, run_id=rid)
    assert ev.kind == "RUN_STARTED"
    assert ev.data == {"threadId": str(tid), "runId": str(rid)}


def test_stream_event_run_finished_constructor() -> None:
    tid, rid = uuid4(), uuid4()
    ev = StreamEvent.run_finished(thread_id=tid, run_id=rid)
    assert ev.kind == "RUN_FINISHED"
    assert ev.data == {"threadId": str(tid), "runId": str(rid)}


def test_stream_event_run_error_constructor() -> None:
    ev = StreamEvent.run_error("boom", code="E_X")
    assert ev.kind == "RUN_ERROR"
    assert ev.data == {"message": "boom", "code": "E_X"}


def test_stream_event_run_error_omits_code_when_none() -> None:
    ev = StreamEvent.run_error("boom")
    assert ev.data == {"message": "boom"}


def test_stream_event_text_message_start_constructor() -> None:
    mid = uuid4()
    ev = StreamEvent.text_message_start(message_id=mid)
    assert ev.kind == "TEXT_MESSAGE_START"
    assert ev.data == {"messageId": str(mid), "role": "assistant"}


def test_stream_event_text_message_content_constructor() -> None:
    mid = uuid4()
    ev = StreamEvent.text_message_content(message_id=mid, delta="hi")
    assert ev.kind == "TEXT_MESSAGE_CONTENT"
    assert ev.data == {"messageId": str(mid), "delta": "hi"}


def test_stream_event_text_message_end_constructor() -> None:
    mid = uuid4()
    ev = StreamEvent.text_message_end(message_id=mid)
    assert ev.kind == "TEXT_MESSAGE_END"
    assert ev.data == {"messageId": str(mid)}


def test_stream_event_tool_call_start_constructor() -> None:
    tcid, pmid = uuid4(), uuid4()
    ev = StreamEvent.tool_call_start(
        tool_call_id=tcid, tool_call_name="search", parent_message_id=pmid,
    )
    assert ev.kind == "TOOL_CALL_START"
    assert ev.data == {
        "toolCallId": str(tcid),
        "toolCallName": "search",
        "parentMessageId": str(pmid),
    }


def test_stream_event_tool_call_args_constructor() -> None:
    tcid = uuid4()
    ev = StreamEvent.tool_call_args(tool_call_id=tcid, delta='{"q":')
    assert ev.kind == "TOOL_CALL_ARGS"
    assert ev.data == {"toolCallId": str(tcid), "delta": '{"q":'}


def test_stream_event_tool_call_end_constructor() -> None:
    tcid = uuid4()
    ev = StreamEvent.tool_call_end(tool_call_id=tcid)
    assert ev.kind == "TOOL_CALL_END"
    assert ev.data == {"toolCallId": str(tcid)}


def test_stream_event_tool_call_constructors_accept_str_tool_call_id() -> None:
    """pydantic-ai surfaces provider-native ids as plain strings (e.g.
    OpenAI's ``"call_abc123"``). The constructors must pass these
    through unchanged so the wire field equals the upstream id verbatim.
    """
    pmid = uuid4()
    start = StreamEvent.tool_call_start(
        tool_call_id="call_abc123",
        tool_call_name="search",
        parent_message_id=pmid,
    )
    args = StreamEvent.tool_call_args(
        tool_call_id="call_abc123", delta='{"q":"x"}',
    )
    end = StreamEvent.tool_call_end(tool_call_id="call_abc123")
    assert start.data["toolCallId"] == "call_abc123"
    assert args.data["toolCallId"] == "call_abc123"
    assert end.data["toolCallId"] == "call_abc123"


def test_ag_ui_encoder_emits_run_started_frame() -> None:
    tid, rid = uuid4(), uuid4()
    frame = AGUIEncoder().encode(StreamEvent.run_started(tid, rid))
    name, data = _parse(frame)
    assert name == "RUN_STARTED"
    assert data == {"threadId": str(tid), "runId": str(rid)}


def test_ag_ui_encoder_emits_run_finished_frame() -> None:
    tid, rid = uuid4(), uuid4()
    frame = AGUIEncoder().encode(StreamEvent.run_finished(tid, rid))
    name, data = _parse(frame)
    assert name == "RUN_FINISHED"
    assert data == {"threadId": str(tid), "runId": str(rid)}


def test_ag_ui_encoder_emits_run_error_frame() -> None:
    frame = AGUIEncoder().encode(StreamEvent.run_error("nope"))
    name, data = _parse(frame)
    assert name == "RUN_ERROR"
    assert data == {"message": "nope"}


def test_ag_ui_encoder_emits_text_message_start_frame() -> None:
    mid = uuid4()
    frame = AGUIEncoder().encode(StreamEvent.text_message_start(mid))
    name, data = _parse(frame)
    assert name == "TEXT_MESSAGE_START"
    assert data == {"messageId": str(mid), "role": "assistant"}


def test_ag_ui_encoder_emits_text_message_content_frame() -> None:
    mid = uuid4()
    frame = AGUIEncoder().encode(
        StreamEvent.text_message_content(mid, "hello"),
    )
    name, data = _parse(frame)
    assert name == "TEXT_MESSAGE_CONTENT"
    assert data == {"messageId": str(mid), "delta": "hello"}


def test_ag_ui_encoder_emits_text_message_end_frame() -> None:
    mid = uuid4()
    frame = AGUIEncoder().encode(StreamEvent.text_message_end(mid))
    name, data = _parse(frame)
    assert name == "TEXT_MESSAGE_END"


def test_ag_ui_encoder_emits_tool_call_start_frame() -> None:
    tcid, pmid = uuid4(), uuid4()
    frame = AGUIEncoder().encode(
        StreamEvent.tool_call_start(tcid, "search", pmid),
    )
    name, data = _parse(frame)
    assert name == "TOOL_CALL_START"
    assert data["toolCallName"] == "search"


def test_ag_ui_encoder_emits_tool_call_args_frame() -> None:
    tcid = uuid4()
    frame = AGUIEncoder().encode(
        StreamEvent.tool_call_args(tcid, '{"q":"x"}'),
    )
    name, data = _parse(frame)
    assert name == "TOOL_CALL_ARGS"
    assert data == {"toolCallId": str(tcid), "delta": '{"q":"x"}'}


def test_ag_ui_encoder_emits_tool_call_end_frame() -> None:
    tcid = uuid4()
    frame = AGUIEncoder().encode(StreamEvent.tool_call_end(tcid))
    name, _data = _parse(frame)
    assert name == "TOOL_CALL_END"


def test_ag_ui_encoder_rejects_unknown_kind() -> None:
    enc = AGUIEncoder()
    with pytest.raises(ValueError, match="unknown StreamEvent kind"):
        enc.encode(StreamEvent(kind="text_delta", data={"text": "hi"}))


def test_ag_ui_encoder_escapes_newlines_in_data() -> None:
    """SSE data lines MUST NOT contain raw \\n — JSON-encode payload."""
    enc = AGUIEncoder()
    mid = uuid4()
    frame = enc.encode(StreamEvent.text_message_content(mid, "a\nb"))
    text = frame.decode("utf-8")
    assert text.count("\n") == 3


def test_stream_event_kind_enum_wire_values() -> None:
    assert StreamEventKind.RUN_STARTED.value == "RUN_STARTED"
    assert StreamEventKind.TEXT_MESSAGE_CONTENT.value == "TEXT_MESSAGE_CONTENT"
    assert StreamEventKind.TOOL_CALL_START.value == "TOOL_CALL_START"


@pytest.mark.asyncio
async def test_streaming_endpoint_streams_events_as_sse() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(
        *, thread_id: UUID, run_id: UUID, message: object, tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        msg_id = uuid4()
        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.text_message_start(message_id=msg_id)
        yield StreamEvent.text_message_content(message_id=msg_id, delta="he")
        yield StreamEvent.text_message_content(message_id=msg_id, delta="llo")
        yield StreamEvent.text_message_end(message_id=msg_id)
        yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body_text = r.text
    for token in (
        "event: RUN_STARTED",
        "event: TEXT_MESSAGE_START",
        "event: TEXT_MESSAGE_CONTENT",
        "event: TEXT_MESSAGE_END",
        "event: RUN_FINISHED",
    ):
        assert token in body_text


@pytest.mark.asyncio
async def test_streaming_endpoint_persists_user_message_before_streaming() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw: object) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.run_finished(uuid4(), uuid4())

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
    with TestClient(app) as c:
        c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
    msgs = await repo.history(th.id, tenant_id=tid)
    assert len(msgs) == 1
    assert msgs[0].role == "user"


@pytest.mark.asyncio
async def test_streaming_endpoint_404_when_thread_missing() -> None:
    repo = InMemoryThreadRepository()

    async def runner(**_kw: object) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.run_finished(uuid4(), uuid4())

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"type": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{uuid4()}/messages", json=body,
            headers={"X-Tenant-Id": str(uuid4())},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_streaming_endpoint_404_cross_tenant() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw: object) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.run_finished(uuid4(), uuid4())

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"type": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages", json=body,
            headers={"X-Tenant-Id": str(other)},
        )
    assert r.status_code == 404

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.streaming.router import (
    StreamEvent,
    build_streaming_router,
)
from pydantic_ai_stateflow.api.streaming.vercel import VercelEncoder
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


def test_vercel_encoder_text_message_content_line() -> None:
    enc = VercelEncoder()
    frame = enc.encode(StreamEvent.text_message_content(uuid4(), "hi"))
    assert frame == b'0:"hi"\n'


def test_vercel_encoder_tool_call_start_line() -> None:
    enc = VercelEncoder()
    tcid = uuid4()
    frame = enc.encode(StreamEvent.tool_call_start(tcid, "search", uuid4()))
    assert frame.startswith(b"b:")
    body = json.loads(frame[2:].decode().rstrip("\n"))
    assert body == {"toolCallId": str(tcid), "toolName": "search"}


def test_vercel_encoder_tool_call_args_line() -> None:
    enc = VercelEncoder()
    tcid = uuid4()
    frame = enc.encode(StreamEvent.tool_call_args(tcid, '{"q":'))
    assert frame.startswith(b"c:")
    body = json.loads(frame[2:].decode().rstrip("\n"))
    assert body == {"toolCallId": str(tcid), "argsTextDelta": '{"q":'}


def test_vercel_encoder_tool_call_end_line() -> None:
    enc = VercelEncoder()
    tcid = uuid4()
    frame = enc.encode(StreamEvent.tool_call_end(tcid))
    assert frame.startswith(b"9:")
    body = json.loads(frame[2:].decode().rstrip("\n"))
    assert body == {"toolCallId": str(tcid), "args": {}}


def test_vercel_encoder_run_finished_line() -> None:
    enc = VercelEncoder()
    frame = enc.encode(StreamEvent.run_finished(uuid4(), uuid4()))
    assert frame == b'd:{"finishReason":"stop"}\n'


def test_vercel_encoder_run_error_line() -> None:
    enc = VercelEncoder()
    frame = enc.encode(StreamEvent.run_error("boom"))
    assert frame == b'3:"boom"\n'


def test_vercel_encoder_drops_run_started() -> None:
    enc = VercelEncoder()
    assert enc.encode(StreamEvent.run_started(uuid4(), uuid4())) == b""


def test_vercel_encoder_drops_text_message_start() -> None:
    enc = VercelEncoder()
    assert enc.encode(StreamEvent.text_message_start(uuid4())) == b""


def test_vercel_encoder_drops_text_message_end() -> None:
    enc = VercelEncoder()
    assert enc.encode(StreamEvent.text_message_end(uuid4())) == b""


@pytest.mark.asyncio
async def test_router_selects_vercel_encoder_via_query() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw: object) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.text_message_content(uuid4(), "hi")
        yield StreamEvent.run_finished(uuid4(), uuid4())

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": [{"kind": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages?protocol=vercel",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    text = r.text
    assert '0:"hi"' in text
    assert 'd:{"finishReason":"stop"}' in text


@pytest.mark.asyncio
async def test_router_skips_empty_frames_from_encoder() -> None:
    """Vercel-dropped kinds must not produce stray blank lines."""
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw: object) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.run_started(uuid4(), uuid4())
        yield StreamEvent.text_message_start(uuid4())
        yield StreamEvent.text_message_content(uuid4(), "hi")

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": []}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages?protocol=vercel",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.text == '0:"hi"\n'


@pytest.mark.asyncio
async def test_router_defaults_to_ag_ui() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw: object) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.text_message_content(uuid4(), "hi")

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": []}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert "event: TEXT_MESSAGE_CONTENT" in r.text


@pytest.mark.asyncio
async def test_router_400_on_unknown_protocol() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw: object) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.run_finished(uuid4(), uuid4())

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": []}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages?protocol=ws",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 400

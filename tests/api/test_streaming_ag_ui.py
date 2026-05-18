from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.streaming.ag_ui import AGUIEncoder
from pydantic_ai_stateflow.api.streaming.router import (
    StreamEvent,
    build_streaming_router,
)
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


def test_ag_ui_encoder_emits_sse_frame_for_text_delta():
    enc = AGUIEncoder()
    frame = enc.encode(StreamEvent(kind="text_delta", data={"text": "hi"}))
    text = frame.decode("utf-8")
    assert text.startswith("event: text_delta\n")
    assert "data: " in text
    assert text.endswith("\n\n")


def test_ag_ui_encoder_emits_done_event():
    enc = AGUIEncoder()
    frame = enc.encode(StreamEvent(kind="done", data={}))
    assert b"event: done" in frame


def test_ag_ui_encoder_escapes_newlines_in_data():
    """SSE data lines MUST NOT contain raw \\n — JSON-encode payload."""
    enc = AGUIEncoder()
    frame = enc.encode(StreamEvent(kind="text_delta", data={"text": "a\nb"}))
    text = frame.decode("utf-8")
    assert text.count("\n") == 3


@pytest.mark.asyncio
async def test_streaming_endpoint_streams_events_as_sse():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(
        *, thread_id, message, tenant_id,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="text_delta", data={"text": "he"})
        yield StreamEvent(kind="text_delta", data={"text": "llo"})
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body_text = r.text
    assert "event: text_delta" in body_text
    assert "event: done" in body_text


@pytest.mark.asyncio
async def test_streaming_endpoint_persists_user_message_before_streaming():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "hello"}]}
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
async def test_streaming_endpoint_404_when_thread_missing():
    repo = InMemoryThreadRepository()

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{uuid4()}/messages", json=body,
            headers={"X-Tenant-Id": str(uuid4())},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_streaming_endpoint_404_cross_tenant():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages", json=body,
            headers={"X-Tenant-Id": str(other)},
        )
    assert r.status_code == 404

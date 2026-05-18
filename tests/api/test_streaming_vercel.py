from __future__ import annotations

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


def test_vercel_encoder_text_delta_line():
    enc = VercelEncoder()
    frame = enc.encode(StreamEvent(kind="text_delta", data={"text": "hi"}))
    assert frame == b'0:"hi"\n'


def test_vercel_encoder_done_line():
    enc = VercelEncoder()
    frame = enc.encode(
        StreamEvent(kind="done", data={"finish_reason": "stop"}),
    )
    text = frame.decode("utf-8")
    assert text.startswith("d:")
    assert text.endswith("\n")


def test_vercel_encoder_tool_call_line():
    enc = VercelEncoder()
    frame = enc.encode(StreamEvent(
        kind="tool_call",
        data={"tool_call_id": "t1", "tool_name": "search", "args": {"q": "x"}},
    ))
    assert frame.startswith(b"9:")


@pytest.mark.asyncio
async def test_router_selects_vercel_encoder_via_query():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="text_delta", data={"text": "hi"})
        yield StreamEvent(kind="done", data={})

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
    assert text.endswith("d:{}\n") or "d:" in text


@pytest.mark.asyncio
async def test_router_defaults_to_ag_ui():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="text_delta", data={"text": "hi"})

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": []}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert "event: text_delta" in r.text


@pytest.mark.asyncio
async def test_router_400_on_unknown_protocol():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": []}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages?protocol=ws",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 400

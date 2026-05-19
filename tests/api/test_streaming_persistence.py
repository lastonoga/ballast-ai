"""F7 — Router auto-persists the assistant reply on TEXT_MESSAGE_END."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.streaming.router import (
    StreamEvent,
    build_streaming_router,
)
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


@pytest.mark.asyncio
async def test_router_persists_assistant_reply_on_text_message_end() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(
        *, thread_id: UUID, run_id: UUID, message: object, tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        mid = uuid4()
        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.text_message_start(message_id=mid)
        yield StreamEvent.text_message_content(message_id=mid, delta="He")
        yield StreamEvent.text_message_content(message_id=mid, delta="llo")
        yield StreamEvent.text_message_end(message_id=mid)
        yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
        assert r.status_code == 200
        # Consume the body to ensure _gen() runs to completion.
        _ = r.text

    msgs = await repo.history(th.id, tenant_id=tid)
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert msgs[1].parts == [{"type": "text", "text": "Hello"}]


@pytest.mark.asyncio
async def test_router_skips_assistant_persist_on_run_error() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(
        *, thread_id: UUID, run_id: UUID, message: object, tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        mid = uuid4()
        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.text_message_start(message_id=mid)
        yield StreamEvent.text_message_content(message_id=mid, delta="Hel")
        yield StreamEvent.run_error("boom")

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
        assert r.status_code == 200
        _ = r.text

    msgs = await repo.history(th.id, tenant_id=tid)
    assert len(msgs) == 1
    assert msgs[0].role == "user"


@pytest.mark.asyncio
async def test_router_handles_tool_only_run_without_text() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(
        *, thread_id: UUID, run_id: UUID, message: object, tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        tcid, pmid = uuid4(), uuid4()
        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.tool_call_start(tcid, "search", pmid)
        yield StreamEvent.tool_call_args(tcid, '{"q":"x"}')
        yield StreamEvent.tool_call_end(tcid)
        yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
        assert r.status_code == 200
        _ = r.text

    msgs = await repo.history(th.id, tenant_id=tid)
    assert len(msgs) == 1
    assert msgs[0].role == "user"


@pytest.mark.asyncio
async def test_router_persists_multiple_assistant_messages_per_run() -> None:
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(
        *, thread_id: UUID, run_id: UUID, message: object, tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        mid1, mid2 = uuid4(), uuid4()
        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.text_message_start(message_id=mid1)
        yield StreamEvent.text_message_content(message_id=mid1, delta="one")
        yield StreamEvent.text_message_end(message_id=mid1)
        yield StreamEvent.text_message_start(message_id=mid2)
        yield StreamEvent.text_message_content(message_id=mid2, delta="two")
        yield StreamEvent.text_message_end(message_id=mid2)
        yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
        assert r.status_code == 200
        _ = r.text

    msgs = await repo.history(th.id, tenant_id=tid)
    assert [m.role for m in msgs] == ["user", "assistant", "assistant"]
    assert msgs[1].parts == [{"type": "text", "text": "one"}]
    assert msgs[2].parts == [{"type": "text", "text": "two"}]

"""F10 — verify client-disconnect propagation on the streaming endpoint.

Investigation summary (recorded for future maintainers):

`httpx.ASGITransport` does NOT simulate a real mid-stream client
disconnect: its ``receive()`` returns ``http.disconnect`` only AFTER the
ASGI app's response is fully complete (it gates the disconnect on
``response_complete.wait()``). And its response stream collects all body
parts into a single in-memory buffer and only yields them once
``app(scope, receive, send)`` returns. Consequently a ``c.stream(...) →
aiter_bytes() → break`` pattern under ASGITransport cannot drive a true
mid-flight disconnect — the app simply never sees one and the
test hangs.

Real-world behaviour (uvicorn / hypercorn): the underlying TCP socket
close triggers ``http.disconnect`` on the ASGI ``receive`` channel.
Starlette's ``Request.is_disconnected()`` consumes that event and
returns ``True``. Relying purely on ``CancelledError`` propagating into
the StreamingResponse generator is NOT reliable across server
implementations either, so we run an explicit poll loop on
``request.is_disconnected()`` concurrently with the agent runner and
cancel the producer when the client goes away.

This test verifies that poll loop end-to-end by patching ``Request``'s
``is_disconnected`` to flip to ``True`` mid-stream and asserting the
runner's ``CancelledError`` cleanup ran.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from pydantic_ai_stateflow.api.streaming import StreamEvent, build_streaming_router
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


@pytest.mark.asyncio
async def test_stream_cancels_runner_on_client_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``request.is_disconnected()`` flips to True, the runner's
    async generator is cancelled and its cleanup runs."""
    from starlette.requests import Request as StarletteRequest

    started = asyncio.Event()
    disconnect = asyncio.Event()
    cleanup_ran = asyncio.Event()

    async def slow_runner(
        *,
        thread_id: UUID,
        run_id: UUID,
        message: _PostMessageBody,
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        del message, tenant_id
        try:
            yield StreamEvent.run_started(thread_id, run_id)
            started.set()
            # Block forever so we can disconnect mid-stream.
            await asyncio.sleep(60)
            yield StreamEvent.run_finished(thread_id, run_id)
        except asyncio.CancelledError:
            cleanup_ran.set()
            raise

    async def fake_is_disconnected(self: StarletteRequest) -> bool:
        return disconnect.is_set()

    monkeypatch.setattr(
        StarletteRequest, "is_disconnected", fake_is_disconnected,
    )

    repo = InMemoryThreadRepository()
    tid = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="a",
        tenant_id=tid,
    )

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=slow_runner),
    )

    transport = httpx.ASGITransport(app=app)

    async def fire_request() -> None:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t",
        ) as c:
            with contextlib.suppress(
                httpx.ReadError, httpx.RemoteProtocolError,
            ):
                await c.post(
                    f"/threads/{thread.id}/messages",
                    headers={"X-Tenant-Id": str(tid)},
                    json={
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                    timeout=5.0,
                )

    req_task = asyncio.create_task(fire_request())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    # Simulate a client disconnect — flip is_disconnected() to True.
    disconnect.set()
    await asyncio.wait_for(cleanup_ran.wait(), timeout=2.0)
    # Let the request task wind down — the producer was cancelled so
    # the generator should return promptly.
    with contextlib.suppress(TimeoutError, Exception):  # noqa: BLE001
        await asyncio.wait_for(req_task, timeout=2.0)
    if not req_task.done():
        req_task.cancel()

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pydantic_ai_stateflow.api.deps import get_tenant_id
from pydantic_ai_stateflow.api.streaming.ag_ui import AGUIEncoder
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository


class StreamEvent(BaseModel):
    """Protocol-neutral streaming event emitted by the agent runner."""

    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class StreamEncoder(Protocol):
    media_type: str

    def encode(self, event: StreamEvent) -> bytes: ...


class _PostMessageBody(BaseModel):
    role: str = "user"
    parts: list[dict[str, Any]] = Field(default_factory=list)


AgentRunner = Callable[..., AsyncIterator[StreamEvent]]

_TenantDep = Depends(get_tenant_id)


def build_streaming_router(
    *,
    thread_repo: ThreadRepository,
    agent_runner: AgentRunner,
    encoder: StreamEncoder | None = None,
    prefix: str = "",
) -> APIRouter:
    """Mount `POST {prefix}/threads/{id}/messages` as an SSE stream.

    `agent_runner` is a callable returning an async iterator of `StreamEvent`s.
    Provide a fake in tests; production wires it to `agent.run_stream(...)` /
    `agent.iter(...)`. The user message is persisted BEFORE the stream starts
    so a client crash mid-stream still leaves the thread consistent.
    """
    router = APIRouter(prefix=prefix)
    enc: StreamEncoder = encoder or AGUIEncoder()

    @router.post("/threads/{thread_id}/messages")
    async def post_message(
        thread_id: UUID,
        body: _PostMessageBody,
        tenant_id: UUID = _TenantDep,
    ) -> StreamingResponse:
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        await thread_repo.add_message(
            thread_id, role=body.role, parts=body.parts, tenant_id=tenant_id,
        )

        async def _gen() -> AsyncIterator[bytes]:
            async for event in agent_runner(
                thread_id=thread_id, message=body, tenant_id=tenant_id,
            ):
                yield enc.encode(event)

        return StreamingResponse(_gen(), media_type=enc.media_type)

    return router

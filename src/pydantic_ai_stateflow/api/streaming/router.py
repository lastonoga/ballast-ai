from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pydantic_ai_stateflow.api.deps import get_tenant_id
from pydantic_ai_stateflow.api.streaming.ag_ui import AGUIEncoder
from pydantic_ai_stateflow.api.streaming.kinds import StreamEventKind
from pydantic_ai_stateflow.api.streaming.vercel import VercelEncoder
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository


class StreamEvent(BaseModel):
    """Protocol-neutral streaming event emitted by the agent runner.

    ``kind`` is a wire string (typically a :class:`StreamEventKind` value).
    ``data`` holds the per-kind payload using AG-UI camelCase field names
    (``threadId``, ``runId``, ``messageId``, ``toolCallId`` …) so encoders can
    serialize without re-mapping.
    """

    kind: str
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def run_started(cls, thread_id: UUID, run_id: UUID) -> StreamEvent:
        return cls(
            kind=StreamEventKind.RUN_STARTED.value,
            data={"threadId": str(thread_id), "runId": str(run_id)},
        )

    @classmethod
    def run_finished(cls, thread_id: UUID, run_id: UUID) -> StreamEvent:
        return cls(
            kind=StreamEventKind.RUN_FINISHED.value,
            data={"threadId": str(thread_id), "runId": str(run_id)},
        )

    @classmethod
    def run_error(cls, message: str, code: str | None = None) -> StreamEvent:
        data: dict[str, Any] = {"message": message}
        if code is not None:
            data["code"] = code
        return cls(kind=StreamEventKind.RUN_ERROR.value, data=data)

    @classmethod
    def text_message_start(
        cls, message_id: UUID, role: str = "assistant",
    ) -> StreamEvent:
        return cls(
            kind=StreamEventKind.TEXT_MESSAGE_START.value,
            data={"messageId": str(message_id), "role": role},
        )

    @classmethod
    def text_message_content(cls, message_id: UUID, delta: str) -> StreamEvent:
        return cls(
            kind=StreamEventKind.TEXT_MESSAGE_CONTENT.value,
            data={"messageId": str(message_id), "delta": delta},
        )

    @classmethod
    def text_message_end(cls, message_id: UUID) -> StreamEvent:
        return cls(
            kind=StreamEventKind.TEXT_MESSAGE_END.value,
            data={"messageId": str(message_id)},
        )

    @classmethod
    def tool_call_start(
        cls,
        tool_call_id: UUID,
        tool_call_name: str,
        parent_message_id: UUID,
    ) -> StreamEvent:
        return cls(
            kind=StreamEventKind.TOOL_CALL_START.value,
            data={
                "toolCallId": str(tool_call_id),
                "toolCallName": tool_call_name,
                "parentMessageId": str(parent_message_id),
            },
        )

    @classmethod
    def tool_call_args(cls, tool_call_id: UUID, delta: str) -> StreamEvent:
        return cls(
            kind=StreamEventKind.TOOL_CALL_ARGS.value,
            data={"toolCallId": str(tool_call_id), "delta": delta},
        )

    @classmethod
    def tool_call_end(cls, tool_call_id: UUID) -> StreamEvent:
        return cls(
            kind=StreamEventKind.TOOL_CALL_END.value,
            data={"toolCallId": str(tool_call_id)},
        )


class StreamEncoder(Protocol):
    media_type: str

    def encode(self, event: StreamEvent) -> bytes: ...


class _PostMessageBody(BaseModel):
    role: str = "user"
    parts: list[dict[str, Any]] = Field(default_factory=list)


AgentRunner = Callable[..., AsyncIterator[StreamEvent]]

_ENCODERS: dict[str, type] = {"ag-ui": AGUIEncoder, "vercel": VercelEncoder}

_TenantDep = Depends(get_tenant_id)
_ProtocolQuery = Query(default="ag-ui")


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

    If `encoder` is supplied it overrides the per-request `?protocol=` query
    param; otherwise the encoder is chosen from `_ENCODERS` by protocol.
    Encoders may return ``b""`` for events they intentionally drop (e.g.
    Vercel has no analog for ``RUN_STARTED``); empty frames are skipped.
    """
    router = APIRouter(prefix=prefix)

    @router.post("/threads/{thread_id}/messages")
    async def post_message(
        thread_id: UUID,
        body: _PostMessageBody,
        tenant_id: UUID = _TenantDep,
        protocol: str = _ProtocolQuery,
    ) -> StreamingResponse:
        if protocol not in _ENCODERS:
            raise HTTPException(
                status_code=400, detail=f"unknown protocol: {protocol}",
            )
        chosen: StreamEncoder = encoder or _ENCODERS[protocol]()
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
                frame = chosen.encode(event)
                if frame:
                    yield frame

        return StreamingResponse(_gen(), media_type=chosen.media_type)

    return router

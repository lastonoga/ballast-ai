from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, runtime_checkable
from uuid import UUID, uuid4

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


# ---------------------------------------------------------------------------
# Typed MessagePart union (F3)
# ---------------------------------------------------------------------------


class _TextPart(BaseModel):
    """Plain text part. Mirrors assistant-ui's `TextMessagePart` shape."""

    type: Literal["text"]
    text: str


class _ToolResultPart(BaseModel):
    """Result of a previously-emitted tool call, sent back by the client."""

    type: Literal["tool-result"]
    tool_call_id: str
    result: Any


class _FilePart(BaseModel):
    """Inline file (image, audio, …) attached to the message."""

    type: Literal["file"]
    data: str
    """Base64-encoded file contents."""
    mime_type: str
    filename: str | None = None


MessagePart = Annotated[
    _TextPart | _ToolResultPart | _FilePart,
    Field(discriminator="type"),
]
"""Discriminated union of allowed parts on `_PostMessageBody.parts`.

Variants follow assistant-ui's `MessagePart` shapes — keep the ``type``
literals in sync with the frontend (see
``examples/notes-app/frontend/RETRO.md``).
"""


def extract_text(parts: list[Any]) -> str:
    """Concatenate all text parts in order. Non-text parts are skipped.

    Accepts both validated `MessagePart` instances and raw dicts (so it
    works on `_PostMessageBody.parts` and on serialized repo rows).
    """
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, _TextPart):
            chunks.append(p.text)
        elif isinstance(p, dict) and p.get("type") == "text":
            text = p.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


class _PostMessageBody(BaseModel):
    role: str = "user"
    parts: list[MessagePart] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentRunner Protocol (F4)
# ---------------------------------------------------------------------------


if TYPE_CHECKING:
    # Forward ref purely for the Protocol signature; runtime resolution not
    # needed since Protocol bodies aren't introspected for kwargs.
    pass


@runtime_checkable
class AgentRunner(Protocol):
    """Adapter that turns a chat message into a stream of AG-UI events.

    ``thread_id``, ``run_id``, and ``tenant_id`` come from the framework;
    ``message`` is the parsed :class:`_PostMessageBody`. Implementations
    should yield canonical
    ``StreamEvent.run_started`` → ``text_message_*`` → ``StreamEvent.run_finished``
    sequences (see :class:`StreamEventKind`).
    """

    def __call__(
        self,
        *,
        thread_id: UUID,
        run_id: UUID,
        message: _PostMessageBody,
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]: ...


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

    `agent_runner` is an :class:`AgentRunner` returning an async iterator of
    `StreamEvent`s. Provide a fake in tests; production wires it to
    :func:`pydantic_ai_stateflow.api.streaming.make_runner`. The user
    message is persisted BEFORE the stream starts so a client crash
    mid-stream still leaves the thread consistent.

    A fresh ``run_id`` is generated per POST and passed to the runner so
    its emitted ``RUN_STARTED`` / ``RUN_FINISHED`` events stay correlated.

    **Assistant reply persistence (F7).** The router observes the event
    stream and auto-persists the assistant message on each
    ``TEXT_MESSAGE_END``. The runner does NOT need to call
    ``repo.add_message`` for the assistant turn. Specifically:

    - ``TEXT_MESSAGE_START`` resets a per-message accumulator
      (multiple messages per run are supported).
    - ``TEXT_MESSAGE_CONTENT`` appends the ``delta``.
    - ``TEXT_MESSAGE_END`` flushes the accumulator as a single
      assistant message via ``thread_repo.add_message``.
    - If the stream errors out (``RUN_ERROR``) before
      ``TEXT_MESSAGE_END``, the partial assistant text is NOT persisted
      (the user message is already persisted before the stream starts).
    - Tool-only runs (no text events) persist nothing extra.

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
            thread_id,
            role=body.role,
            parts=[p.model_dump() for p in body.parts],
            tenant_id=tenant_id,
        )
        run_id = uuid4()

        async def _gen() -> AsyncIterator[bytes]:
            accumulated_text = ""
            assistant_message_id: UUID | None = None
            async for event in agent_runner(
                thread_id=thread_id,
                run_id=run_id,
                message=body,
                tenant_id=tenant_id,
            ):
                if event.kind == StreamEventKind.TEXT_MESSAGE_START.value:
                    raw_mid = event.data.get("messageId")
                    assistant_message_id = (
                        UUID(raw_mid) if isinstance(raw_mid, str) else None
                    )
                    accumulated_text = ""
                elif event.kind == StreamEventKind.TEXT_MESSAGE_CONTENT.value:
                    delta = event.data.get("delta", "")
                    if isinstance(delta, str):
                        accumulated_text += delta
                elif event.kind == StreamEventKind.TEXT_MESSAGE_END.value:
                    if assistant_message_id is not None:
                        await thread_repo.add_message(
                            thread_id,
                            role="assistant",
                            parts=[{"type": "text", "text": accumulated_text}],
                            tenant_id=tenant_id,
                        )
                    # Reset for any subsequent message in the same run.
                    assistant_message_id = None
                    accumulated_text = ""
                frame = chosen.encode(event)
                if frame:
                    yield frame

        return StreamingResponse(_gen(), media_type=chosen.media_type)

    return router

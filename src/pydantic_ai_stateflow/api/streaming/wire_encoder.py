"""Pluggable wire-format encoders for the durable streaming branch.

The durable streaming router reads ``ThreadEvent`` rows from the
event log + ``EventNotification``s from the signal channel and turns
them into SSE chunks bound for the client. The exact wire format is
delegated here so apps can plug in different transports:

  - ``VercelAIWireEncoder``  (default) — Vercel AI SDK v6 over SSE,
    consumed by ``useChat`` on the frontend.
  - ``AGUIWireEncoder``     — AG-UI canonical events. (TODO — sketch
    only.)
  - app-specific            — implement the ``WireEncoder`` Protocol.

The default ``ThreadEvent.kind`` set the framework emits is small and
neutral:

  - ``start``       — workflow began (payload may carry the user
    prompt for diagnostics).
  - ``text-delta``  — assistant produced text. ``payload["text"]`` is
    the delta (or the full string for MVP).
  - ``tool-call``   — model requested a tool. Payload TBD.
  - ``tool-result`` — tool body returned. Payload TBD.
  - ``done``        — terminal; SSE consumer SHOULD close after this.
  - ``error``       — unrecoverable run failure; payload has details.

Encoders SHOULD be tolerant of unknown ``kind`` values (skip silently)
so adding new event kinds doesn't break older clients on rolling
deploys.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID

    from pydantic_ai_stateflow.persistence.events.domain import ThreadEvent


@runtime_checkable
class WireEncoder(Protocol):
    """Transform ``ThreadEvent`` rows into wire-format bytes.

    The encoder is stateful within one SSE response — implementations
    MAY buffer cross-event state (e.g. text-delta accumulation for
    wire formats that prefer one message-id grouping per assistant
    turn). Each SSE response gets its own encoder instance via the
    framework's ``encoder_factory``.
    """

    def content_type(self) -> str:
        """Media type the SSE response advertises (usually
        ``text/event-stream``)."""
        ...

    def initial_events(self, *, thread_id: UUID) -> Iterable[bytes]:
        """Wire-format prelude (e.g. SSE ``start`` event).

        Emitted once at the top of the response, BEFORE replay or
        live tail. Empty iterable is fine for protocols that don't
        need a preamble.
        """
        ...

    def encode_event(self, event: ThreadEvent) -> Iterable[bytes]:
        """Encode one persisted ``ThreadEvent`` into 0-N wire chunks.

        Implementations SHOULD return an empty iterable for unknown
        event kinds (forward-compatibility).
        """
        ...

    def finalize(self) -> Iterable[bytes]:
        """Wire-format epilogue (e.g. SSE ``[DONE]`` sentinel).

        Emitted exactly once when the durable run completes (after
        the ``done`` ``ThreadEvent``) or the SSE consumer disconnects
        cleanly.
        """
        ...


def _sse(data: str, *, event_id: int | None = None) -> bytes:
    """Format a single SSE chunk.

    ``id:`` lines drive ``EventSource.lastEventId`` on the browser
    side — that's the resume key our router reads back on reconnect.
    """
    parts: list[str] = []
    if event_id is not None:
        parts.append(f"id: {event_id}")
    parts.append(f"data: {data}")
    return ("\n".join(parts) + "\n\n").encode()


class VercelAIWireEncoder:
    """Default encoder — Vercel AI SDK v6 SSE wire format.

    Conservative MVP: emits ``start`` + ``text-delta`` (using the
    framework's neutral payloads) + ``finish`` + ``[DONE]``. Tool-
    call / approval-card encoding lands when the durable workflow
    starts emitting real ``tool-call`` / ``tool-result`` events
    (currently it only emits ``start``, ``text-delta``, ``done``).
    """

    name: ClassVar[str] = "vercel-ai-v6"

    def content_type(self) -> str:
        return "text/event-stream"

    def initial_events(self, *, thread_id: UUID) -> Iterable[bytes]:
        # Vercel AI SDK v6 ``start`` event opens the response.
        yield _sse(json.dumps({
            "type": "start",
            "messageMetadata": {"thread_id": str(thread_id)},
        }))

    def encode_event(self, event: ThreadEvent) -> Iterable[bytes]:
        if event.kind == "text-delta":
            text = str(event.payload.get("text", ""))
            yield _sse(
                json.dumps({"type": "text-delta", "delta": text}),
                event_id=event.seq,
            )
        elif event.kind == "error":
            yield _sse(
                json.dumps({
                    "type": "error",
                    "errorText": str(event.payload.get("message", "error")),
                }),
                event_id=event.seq,
            )
        elif event.kind == "done":
            yield _sse(
                json.dumps({"type": "finish"}),
                event_id=event.seq,
            )
        # ``start`` event from workflow doesn't need a wire emission
        # (we already sent our own start in ``initial_events``).
        # Unknown kinds → silent skip (forward-compat).
        return

    def finalize(self) -> Iterable[bytes]:
        # Vercel AI SDK terminates the stream with ``[DONE]`` sentinel.
        yield b"data: [DONE]\n\n"


# ``Any`` import kept for forward-references to encoder configuration
# payloads; mypy was complaining about the unused import.
_ = Any


__all__ = [
    "VercelAIWireEncoder",
    "WireEncoder",
]

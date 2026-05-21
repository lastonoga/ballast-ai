"""Custom thread events — typed factory for sending app-defined data
through the long-lived SSE on ``GET /threads/{id}/events``.

## Concept

Apps declare event types as module-level constants paired with a
pydantic data schema::

    class BrainstormProgress(BaseModel):
        step: str
        status: Literal["running", "ok", "failed"]
        note: str | None = None

    BRAINSTORM_PROGRESS = ThreadEventType(
        "brainstorm-progress", BrainstormProgress,
    )

Workflows emit one-shot events or open streaming sessions::

    # one-shot — appears in the thread once
    await BRAINSTORM_PROGRESS.emit(
        broadcaster, thread_id,
        BrainstormProgress(step="diverge", status="ok"),
    )

    # streaming — same UI message mutates across updates
    async with BRAINSTORM_PROGRESS.stream(broadcaster, thread_id) as s:
        await s.update(BrainstormProgress(step="diverge", status="running"))
        ...
        await s.update(BrainstormProgress(step="diverge", status="ok"))
        await s.update(BrainstormProgress(step="converge", status="running"))

Frontends register a custom assistant-ui ``MessagePart`` paired by the
wire name (``data-<event_type.name>``)::

    export const BrainstormProgressPart = makeAssistantMessagePart({
      type: "data-brainstorm-progress",   // matches ThreadEventType.part_type
      render: ({ data }) => <ProgressRow step={data.step} status={data.status} />,
    });

## Delivery modes

* **persistent** (default) — the event is written as a message into
  ``thread_repo``. Page reload / ``historyAdapter.load()`` sees it
  too.

* **transient** (``persistent=False``) — live signal only. The UI
  appends the event to its in-memory messages but ``thread_repo`` is
  not touched, so a reload makes it disappear. Useful for toast-like
  progress that doesn't belong in the permanent timeline.

## Streaming semantics

A ``ThreadEventStream`` reuses one stable ``message_id`` across
updates. ``thread_repo.upsert_message`` replaces the previous parts
in place (persistent mode); the SSE signal carries the FULL updated
message, so frontends keyed by id replace rather than append. Net
effect: one UI-visible event that animates as the workflow runs.

## Wire format

No new wire type. The existing ``message-added`` event kind on the
event log carries the message; the part inside it is
``{"type": "data-<name>", "data": {...}, "state": "done"}``. The
``state`` field follows assistant-ui's convention for completed
message parts.

The frontend SSE handler in ``runtime-provider.tsx`` already calls
``setMessages`` on ``message-added`` — extended to replace by id
when the message already exists in the chat so streaming updates
work without an additional ``message-updated`` event kind.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel

from pydantic_ai_stateflow.persistence.events.repository import (
    EventLogRepository,
)
from pydantic_ai_stateflow.runtime.event_stream import (
    EventNotification,
    EventStream,
    thread_channel,
)

if TYPE_CHECKING:
    from types import TracebackType

    from pydantic_ai_stateflow.persistence.thread.repository import (
        ThreadRepository,
    )

_log = logging.getLogger(__name__)

DataT = TypeVar("DataT", bound=BaseModel)


class ThreadEventBroadcaster:
    """Low-level I/O layer for custom thread events.

    Apps don't typically call this directly — they declare a
    ``ThreadEventType`` constant and call its ``emit`` / ``stream``
    methods which delegate here. Exposed publicly so framework code
    that wants to bypass the typed factory (e.g. dynamic event types
    from configuration) can use it.
    """

    def __init__(
        self,
        *,
        thread_repo: ThreadRepository,
        event_log: EventLogRepository,
        event_stream: EventStream | None = None,
    ) -> None:
        self._thread_repo = thread_repo
        self._event_log = event_log
        self._event_stream = event_stream

    async def emit_raw(
        self,
        thread_id: UUID,
        *,
        part: dict[str, Any],
        message_id: str | None = None,
        persistent: bool = True,
        role: str = "assistant",
    ) -> str:
        """Emit a single custom message part to ``thread_id``.

        Returns the ``message_id`` that was used — either the one
        supplied by the caller or a freshly-minted UUID4. Streaming
        callers pass the SAME id on subsequent calls to overwrite the
        previous part.

        ``part`` is the assistant-ui message-part dict — e.g.
        ``{"type": "data-progress", "data": {...}, "state": "done"}``.
        """
        msg_id = message_id or str(uuid4())

        if persistent:
            msg = await self._thread_repo.upsert_message(
                thread_id, id=msg_id, role=role, parts=[part],
            )
            parts_for_signal = msg.parts
            role_for_signal = msg.role
        else:
            # Transient: skip repo write, signal carries the part
            # directly so the UI still gets to render it.
            parts_for_signal = [part]
            role_for_signal = role

        ev = await self._event_log.append(
            thread_id=thread_id,
            kind="message-added",
            payload={
                "id": msg_id,
                "role": role_for_signal,
                "parts": parts_for_signal,
                "transient": not persistent,
            },
        )
        if self._event_stream is not None:
            await self._event_stream.publish(
                thread_channel(thread_id),
                EventNotification(thread_id=thread_id, seq=ev.seq),
            )
        return msg_id


class ThreadEventType(Generic[DataT]):
    """Typed declaration of a custom thread event.

    ``name`` defines the wire string (``data-<name>``) that pairs
    with a frontend ``makeAssistantMessagePart`` registration.
    ``data_schema`` is the pydantic model the data payload must
    satisfy — so ``await EVT.emit(broadcaster, thread_id, data)``
    is type-checked end-to-end.
    """

    def __init__(self, name: str, data_schema: type[DataT]) -> None:
        if not name:
            raise ValueError("event type name must be non-empty")
        if "/" in name or " " in name:
            raise ValueError(
                f"event type name {name!r} contains illegal characters",
            )
        self.name = name
        self.data_schema = data_schema

    @property
    def part_type(self) -> str:
        """Wire string used inside the message part. Frontend custom
        part registration must match this exactly: ``data-<name>``."""
        return f"data-{self.name}"

    async def emit(
        self,
        broadcaster: ThreadEventBroadcaster,
        thread_id: UUID,
        data: DataT,
        *,
        message_id: str | None = None,
        persistent: bool = True,
    ) -> str:
        """One-shot emit. Returns the ``message_id`` used.

        For multiple updates of the same logical event (e.g. a
        progress bar that ticks through stages), prefer
        :meth:`stream` — its context manager keeps the message_id
        stable for you.
        """
        if not isinstance(data, self.data_schema):
            raise TypeError(
                f"event {self.name!r}: data must be {self.data_schema.__name__}, "
                f"got {type(data).__name__}",
            )
        part = {
            "type": self.part_type,
            "data": data.model_dump(mode="json"),
            "state": "done",
        }
        return await broadcaster.emit_raw(
            thread_id,
            part=part,
            message_id=message_id,
            persistent=persistent,
        )

    def stream(
        self,
        broadcaster: ThreadEventBroadcaster,
        thread_id: UUID,
        *,
        persistent: bool = True,
        message_id: str | None = None,
    ) -> ThreadEventStream[DataT]:
        """Open a streaming session for this event type.

        Each ``.update(data)`` overwrites the same message (persistent
        mode) or re-fires a transient signal with the same id
        (transient mode). Use as an async context manager — there's
        no explicit close work, but the context boundary marks the
        logical session end for trace/debug clarity::

            async with EVT.stream(broadcaster, thread_id) as s:
                await s.update(data_v1)
                ...
                await s.update(data_v2)
        """
        return ThreadEventStream(
            event_type=self,
            broadcaster=broadcaster,
            thread_id=thread_id,
            message_id=message_id or str(uuid4()),
            persistent=persistent,
        )


class ThreadEventStream(Generic[DataT]):
    """Streaming session — same ``message_id`` reused across updates.

    Construct via :meth:`ThreadEventType.stream`, NOT directly. The
    stream object is mostly an ergonomic wrapper: it remembers the
    bound type, broadcaster, thread, and message id so callers don't
    pass them on every ``update``.

    Re-entrancy: a stream is NOT safe to share across concurrent
    tasks. If you need parallel writers, give each its own stream
    (and thus its own message id).
    """

    def __init__(
        self,
        *,
        event_type: ThreadEventType[DataT],
        broadcaster: ThreadEventBroadcaster,
        thread_id: UUID,
        message_id: str,
        persistent: bool,
    ) -> None:
        self._event_type = event_type
        self._broadcaster = broadcaster
        self._thread_id = thread_id
        self._message_id = message_id
        self._persistent = persistent
        self._updates: int = 0

    @property
    def message_id(self) -> str:
        """The stable message_id this stream uses across updates."""
        return self._message_id

    @property
    def updates(self) -> int:
        """How many ``update(...)`` calls have been emitted so far."""
        return self._updates

    async def update(self, data: DataT) -> None:
        """Emit (or overwrite) the latest snapshot of this event."""
        await self._event_type.emit(
            self._broadcaster,
            self._thread_id,
            data,
            message_id=self._message_id,
            persistent=self._persistent,
        )
        self._updates += 1

    async def __aenter__(self) -> ThreadEventStream[DataT]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Nothing to clean up — persistence is per-update.
        # We deliberately don't auto-mark "completed" because the
        # final ``update`` typically carries the terminal status
        # itself (e.g. ``status="ok"``).
        del exc_type, exc, tb


__all__ = [
    "ThreadEventBroadcaster",
    "ThreadEventStream",
    "ThreadEventType",
]

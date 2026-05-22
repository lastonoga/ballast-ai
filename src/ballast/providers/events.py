"""EventsProvider — wires the app's :class:`EventLogRepository` and
:class:`EventStream` onto Ballast and connects the framework's default
signal handlers (which turn :data:`message_added` /
:data:`helper_thread_created` into log + publish writes).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from ballast.durable import Durable
from ballast.events import helper_thread_created, message_added
from ballast.runtime.event_stream import EventNotification, thread_channel

if TYPE_CHECKING:
    from ballast.app import Ballast
    from ballast.persistence.events.repository import EventLogRepository
    from ballast.persistence.thread.domain import Message
    from ballast.runtime.event_stream import EventStream


class EventsProvider:
    """Set the event-log repository + in-process event stream on the
    :class:`Engine`, and connect the framework's default signal handlers."""

    def __init__(
        self,
        event_log: "EventLogRepository",
        event_stream: "EventStream",
    ) -> None:
        self._event_log = event_log
        self._event_stream = event_stream

    def register(self, ballast: "Ballast") -> None:
        ballast._set_event_log(self._event_log)
        ballast._set_event_stream(self._event_stream)
        # ``Signal.connect`` is idempotent, so re-registering across
        # fixture rebuilds doesn't double-fire the defaults.
        message_added.connect(_default_message_added)
        helper_thread_created.connect(_default_helper_thread_created)


@Durable.step()
async def _default_message_added(
    sender: Any,  # noqa: ARG001 — receiver contract requires positional sender
    *,
    thread_id: UUID,
    message: "Message",
    **_: Any,
) -> None:
    """Default handler for :data:`ballast.events.message_added`.

    Appends a ``message-added`` event to the durable log and pushes
    a wake-up notification onto the event stream so any SSE consumer
    tailing the thread receives the new message live.

    Wrapped in ``@Durable.step()`` so that callers invoking
    ``thread_repo.add_message`` from inside a ``@Durable.workflow`` body
    do not double-write on crash recovery — DBOS memoises the step's
    completion by name + args.
    """
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    engine = get_ballast()
    ev = await engine.event_log.append(
        thread_id=thread_id,
        kind="message-added",
        payload={
            "id": message.id,
            "role": message.role,
            "parts": message.parts,
        },
    )
    await engine.event_stream.publish(
        thread_channel(thread_id),
        EventNotification(thread_id=thread_id, seq=ev.seq),
    )


@Durable.step()
async def _default_helper_thread_created(
    sender: Any,  # noqa: ARG001
    *,
    parent_thread_id: UUID,
    helper_thread_id: UUID,
    helper_agent_name: str,
    helper_metadata: dict[str, Any],
    **_: Any,
) -> None:
    """Default handler for :data:`ballast.events.helper_thread_created`.

    Emits a ``thread-created`` event into the parent thread's event log
    so a UI tailing ``GET /threads/{id}/events`` refreshes its thread
    list without F5. Memoised across replays via ``@Durable.step``.
    """
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    engine = get_ballast()
    ev = await engine.event_log.append(
        thread_id=parent_thread_id,
        kind="thread-created",
        payload={
            "thread_id": str(helper_thread_id),
            "agent": helper_agent_name,
            "metadata": helper_metadata,
        },
    )
    await engine.event_stream.publish(
        thread_channel(parent_thread_id),
        EventNotification(thread_id=parent_thread_id, seq=ev.seq),
    )


__all__ = ["EventsProvider"]

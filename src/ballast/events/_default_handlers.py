"""Default ``Signal`` handlers connected by :meth:`Ballast.with_events`.

These handlers wire the framework's "messages go through signals"
contract — they take a signal payload, write the durable side-effect
(append an event-log entry, publish an event_stream notification), and
return. Apps can disconnect any of them and replace with a custom
receiver if they want different routing.

  - ``_default_message_added`` — turns ``message_added`` into a
    ``message-added`` event-log row + ``event_stream.publish`` wake-up.
  - ``_default_helper_thread_created`` — emits a ``thread-created``
    event into the parent thread when a HITL helper opens a side
    conversation.

Pattern progress (``data-*`` UI cards) bypasses signals entirely and
writes via :class:`ThreadEventBroadcaster` directly — see the
pattern's own ``default_chat_router`` for the call site.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from ballast.durable import Durable
from ballast.runtime.event_stream import EventNotification, thread_channel

if TYPE_CHECKING:
    from ballast.persistence.thread.domain import Message


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


def connect_default_handlers() -> None:
    """Idempotently connect the framework's default signal handlers.

    Called once from :meth:`Ballast.with_events` so the framework's
    out-of-the-box behaviour is in place. ``Signal.connect`` is
    idempotent, so multiple calls (e.g. across fixture rebuilds) don't
    double-fire.
    """
    from ballast.events import (  # noqa: PLC0415
        helper_thread_created,
        message_added,
    )

    message_added.connect(_default_message_added)
    helper_thread_created.connect(_default_helper_thread_created)


__all__ = [
    "_default_helper_thread_created",
    "_default_message_added",
    "connect_default_handlers",
]

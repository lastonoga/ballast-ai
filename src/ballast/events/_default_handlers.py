"""Default ``Signal`` handlers connected by :meth:`Ballast.with_events`.

These three handlers wire the framework's "messages go through signals"
contract — they take a signal payload, write the durable side-effect
(persist a row, append an event-log entry, publish an event_stream
notification), and return. Apps can disconnect any of them and replace
with a custom receiver if they want different routing.

  - ``_default_chat_message_requested`` — turns a ``chat_message_requested``
    payload into a ``thread_repo.add_message`` (or ``upsert_message`` if
    ``message_id`` is supplied). The repo itself fires ``message_added``.
  - ``_default_message_added`` — turns ``message_added`` into a
    ``message-added`` event-log row + ``event_stream.publish`` wake-up.
  - ``_default_helper_thread_created`` — emits a ``thread-created``
    event into the parent thread when a HITL helper opens a side
    conversation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from ballast.durable import Durable
from ballast.runtime.event_stream import EventNotification, thread_channel

if TYPE_CHECKING:
    from ballast.persistence.thread.domain import Message


async def _default_chat_message_requested(
    sender: Any,  # noqa: ARG001
    *,
    thread_id: UUID,
    text: str | None = None,
    parts: list[dict[str, Any]] | None = None,
    message_id: str | None = None,
    **_: Any,
) -> None:
    """Default handler for :data:`ballast.events.chat_message_requested`.

    ``parts`` (when supplied) takes precedence over ``text``; otherwise
    falls back to one trivial ``{type:text, text, state:done}`` part.

    Routing:

    * ``message_id is None`` (default) → ``thread_repo.add_message`` —
      always a fresh row.
    * ``message_id is set`` → ``thread_repo.upsert_message(id=...)`` —
      replaces parts in place. Callers reuse the same id across
      successive emits to get one mutating chat row instead of N
      stacked messages (e.g. a branch spinner that flips to a check).

    NOT wrapped in ``@Durable.step`` — ``ThreadRepository.add_message``
    /``upsert_message`` fire their own ``message_added`` whose default
    handler IS a step, which is where the durability boundary actually
    matters. Nesting steps here would just add noise to the workflow log.
    """
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    final_parts = parts or [
        {"type": "text", "text": text or "", "state": "done"},
    ]
    repo = get_ballast().thread_repo
    if message_id is None:
        await repo.add_message(
            thread_id, role="assistant", parts=final_parts,
        )
    else:
        await repo.upsert_message(
            thread_id, id=message_id, role="assistant", parts=final_parts,
        )


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
    """Idempotently connect the three default signal handlers.

    Called once from :meth:`Ballast.with_events` so the framework's
    out-of-the-box behaviour is in place. ``Signal.connect`` is
    idempotent, so multiple calls (e.g. across fixture rebuilds) don't
    double-fire.
    """
    from ballast.events import (  # noqa: PLC0415
        chat_message_requested,
        helper_thread_created,
        message_added,
    )

    chat_message_requested.connect(_default_chat_message_requested)
    message_added.connect(_default_message_added)
    helper_thread_created.connect(_default_helper_thread_created)


__all__ = [
    "_default_chat_message_requested",
    "_default_helper_thread_created",
    "_default_message_added",
    "connect_default_handlers",
]

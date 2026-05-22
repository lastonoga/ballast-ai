"""Ballast's in-process signal bus.

Two built-in signals — :data:`message_added` and
:data:`helper_thread_created` — wire the framework's "I just wrote
something" emitters to its "publish to the event log / push to the SSE
stream" defaults so callers don't have to re-implement the
add-message + log + publish dance every time.

Payload contracts
-----------------

``message_added(sender=repo, *, thread_id: UUID, message: Message)``
    Fires AFTER a :class:`ThreadRepository.add_message` or
    :class:`ThreadRepository.upsert_message` successfully writes a
    message row. ``sender`` is the repository instance that produced
    the write; ``message`` is the persisted :class:`Message`.

``helper_thread_created(sender=workflow, *, parent_thread_id: UUID,``
``helper_thread_id: UUID, helper_agent_name: str,``
``helper_metadata: dict)``
    Fires when a HITL flow spawns a helper thread that the parent
    thread's UI should learn about. ``sender`` is the HITL workflow
    module / instance that opened the thread.

Default handlers (in :class:`ballast.providers.events.EventsProvider`)
turn each signal into the durable log append + event-stream publish
that callers used to write inline.
"""

from __future__ import annotations

from ballast.events.adapters import (
    route_to_thread_as_data,
    route_to_thread_as_text,
)
from ballast.events.signals import Signal, receiver

# ── Built-in signals ────────────────────────────────────────────────────

message_added: Signal = Signal("message_added")
"""Emitted after a thread repo appends or upserts a message.

Payload: ``sender=repo, *, thread_id: UUID, message: Message``."""

helper_thread_created: Signal = Signal("helper_thread_created")
"""Emitted when a HITL flow opens a helper thread for the user.

Payload: ``sender=workflow, *, parent_thread_id: UUID,
helper_thread_id: UUID, helper_agent_name: str, helper_metadata: dict``."""


__all__ = [
    "Signal",
    "helper_thread_created",
    "message_added",
    "receiver",
    "route_to_thread_as_data",
    "route_to_thread_as_text",
]

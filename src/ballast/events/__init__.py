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

Default handlers (in :mod:`ballast.events._default_handlers`,
connected by :meth:`Ballast.with_events`) turn each signal into the
durable log append + event-stream publish that callers used to write
inline.

Pattern progress (``data-*`` UI cards rendered live as workflows run)
no longer goes through a signal — patterns call the engine's
:class:`ThreadEventBroadcaster` directly. That removes three hops of
indirection (no more ``chat_message_requested`` → ``add_message`` →
``message_added`` chain for the pattern-progress path) and keeps the
``message_added`` signal narrowly scoped to actual chat-turn writes
(user messages, assistant turns, HITL openings).
"""

from __future__ import annotations

from ballast.events.context import progress_thread_var, progress_to_thread
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
    "progress_thread_var",
    "progress_to_thread",
    "receiver",
]

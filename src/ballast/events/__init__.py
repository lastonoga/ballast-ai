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

chat_message_requested: Signal = Signal("chat_message_requested")
"""Request to append (or update) an assistant chat message on a thread.

Payload: ``sender=anything, *, thread_id: UUID, text: str | None = None,
parts: list[dict] | None = None, message_id: str | None = None``.

The default handler (connected by :class:`EventsProvider` at app
startup) routes through the thread repo:

* ``message_id is None`` → ``add_message`` (always a fresh row;
  default for one-shot narration).
* ``message_id is not None`` → ``upsert_message(id=message_id, ...)`` —
  the row's parts are REPLACED in place. Reuse the same id across
  successive emits to get a single mutating chat row (e.g. a
  "branch enqueued" spinner that transitions to a "branch
  completed" check on the same line instead of stacking up).

Either way the repo self-emits :data:`message_added` afterwards, so
the event log + SSE chain fires once per call regardless of whether
this was an insert or an update.

When ``parts`` is supplied it OVERRIDES the trivial ``[{type:text,
text, state:done}]`` construction so callers can post typed data
parts (``[{type: "data-foo", data: {...}, state: "done"}]``) for
custom UI rendering.

Existing as a signal (not just a function call) so:

  - All "append a message" intents flow through ONE pluggable
    primitive (apps can intercept for audit, filtering, rewriting).
  - The handler is connected at module-load time on the framework
    side, so it fires reliably from any execution context
    (durable-workflow body, queue worker, HTTP handler, …) without
    each caller having to register its own closure."""


__all__ = [
    "Signal",
    "chat_message_requested",
    "helper_thread_created",
    "message_added",
    "progress_thread_var",
    "progress_to_thread",
    "receiver",
]

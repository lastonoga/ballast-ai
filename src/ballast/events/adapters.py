"""Connect pattern-progress :class:`Signal` instances to common destinations.

Patterns (e.g. :data:`ballast.patterns.divergent_convergent.events.divergent_convergent_progress`)
emit typed events as they progress. Apps decide WHERE those events
go by connecting handlers — these helpers wrap the common cases so
the call site stays a single line.

## Typical use

::

    from ballast.events.adapters import route_to_thread_as_text
    from ballast.patterns.divergent_convergent.events import (
        divergent_convergent_progress,
    )

    @Durable.workflow()
    async def my_flow(task):
        disconnect = route_to_thread_as_text(
            divergent_convergent_progress, thread_id=task.parent_thread_id,
        )
        try:
            result = await _pattern.run(...)
        finally:
            disconnect()

The handler captures ``thread_id`` (a workflow input → deterministic
across replay) so connect/disconnect is replay-safe. Connecting the
same callable twice is a no-op (``Signal.connect`` is idempotent), so
nested or repeated wiring also stays clean.

## Two flavours of thread routing

* :func:`route_to_thread_as_text` — formats events into human-readable
  strings + appends them as plain assistant text messages. Zero
  frontend work, looks like ``say()`` from the brainstorm flow.

* :func:`route_to_thread_as_data` — emits typed data parts
  (``{type: "data-<event.type>", data: ..., state: "done"}``).
  Frontend registers a renderer per event type via assistant-ui's
  ``makeAssistantDataUI({name: "branch-completed", render: ...})``
  for rich per-event UI (spinners, badges, custom layout). Falls
  back to nothing in chat when no renderer is registered, so the
  app must wire the FE side for users to see anything.

Apps that need other destinations (Slack, Logfire, Prometheus, custom
audit) connect their own receivers directly via
:func:`ballast.events.receiver`; this module just bundles the common
"to a thread" cases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel

if TYPE_CHECKING:
    from ballast.events.signals import Signal


def route_to_thread_as_text(
    signal: "Signal",
    *,
    thread_id: UUID,
    format_fn: Callable[[BaseModel], str] | None = None,
) -> Callable[[], None]:
    """Append each event as a plain assistant chat message on ``thread_id``.

    Returns a callable that disconnects the handler — call it in a
    ``finally`` block (or rely on the workflow exiting naturally; the
    Signal will keep the handler registered across the process
    lifetime if you don't).

    ``format_fn`` receives the event model and returns the string to
    post. Default formatter renders ``"<event.type>: <model.dict>"``
    minus the redundant ``type`` field — readable enough for dev and
    chat-history scanning. Returning ``""`` (empty string) from
    ``format_fn`` SKIPS the message — handy for suppressing low-signal
    events while still keeping handlers wired.
    """
    fmt = format_fn or _default_text_format

    async def handler(
        sender: Any,  # noqa: ARG001 — receiver contract
        *,
        event: BaseModel,
        **_: Any,
    ) -> None:
        from ballast.runtime.engine import get_ballast  # noqa: PLC0415

        text = fmt(event)
        if not text:
            return  # caller signalled "ignore this event"
        await get_ballast().thread_repo.add_message(
            thread_id,
            role="assistant",
            parts=[{"type": "text", "text": text, "state": "done"}],
        )

    signal.connect(handler)
    return lambda: signal.disconnect(handler)


def route_to_thread_as_data(
    signal: "Signal",
    *,
    thread_id: UUID,
    part_type_prefix: str = "data",
) -> Callable[[], None]:
    """Emit each event as a typed data part on a fresh assistant message.

    Each event becomes one chat row with a single part::

        {"type": "<prefix>-<event.type>", "data": event.model_dump(),
         "state": "done"}

    Frontends register renderers per ``event.type`` via assistant-ui's
    ``makeAssistantDataUI({name: "..."})``. Unrendered parts simply
    don't appear in the chat (assistant-ui drops unknown data parts
    silently).
    """

    async def handler(
        sender: Any,  # noqa: ARG001
        *,
        event: BaseModel,
        **_: Any,
    ) -> None:
        from ballast.runtime.engine import get_ballast  # noqa: PLC0415

        event_type = getattr(event, "type", event.__class__.__name__)
        await get_ballast().thread_repo.add_message(
            thread_id,
            role="assistant",
            parts=[{
                "type": f"{part_type_prefix}-{event_type}",
                "data": event.model_dump(mode="json"),
                "state": "done",
            }],
        )

    signal.connect(handler)
    return lambda: signal.disconnect(handler)


def _default_text_format(event: BaseModel) -> str:
    """Render an event as ``"<type>: k1=v1, k2=v2"``.

    Best-effort default for ``route_to_thread_as_text`` when the app
    hasn't supplied its own ``format_fn``. Strips the ``type`` field
    from the value dump (it's the prefix) and joins the rest as
    comma-separated kwargs.
    """
    event_type = getattr(event, "type", event.__class__.__name__)
    data = event.model_dump()
    data.pop("type", None)
    if not data:
        return event_type
    pairs = ", ".join(f"{k}={v!r}" for k, v in data.items())
    return f"{event_type}: {pairs}"


__all__ = [
    "route_to_thread_as_data",
    "route_to_thread_as_text",
]

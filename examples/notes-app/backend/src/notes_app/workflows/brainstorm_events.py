"""Typed events emitted by the brainstorm workflow.

Same shape as the framework's pattern-event modules (e.g.
``ballast.patterns.divergent_convergent.events``):

  - One pydantic model per observable event (``BrainstormChose``,
    ``BrainstormSaved``, ``BrainstormCancelled``, ``BrainstormTimedOut``).
  - One module-level :class:`Signal` (``brainstorm_progress``) that
    carries any of those events as ``event=...`` kwarg.
  - One default chat-routing handler, auto-connected at module load:
    reads :data:`progress_thread_var`, formats the event into a
    chat line, fires :data:`chat_message_requested`.
  - One :func:`format_for_chat` helper for the rendering.

This lets the workflow stay pure (just ``brainstorm_progress.send(
event=BrainstormChose(...))``) and gives observers a typed channel
they can subscribe to for analytics / metrics / rich-UI rendering on
the frontend.

To customise the chat narration:

* **opt out** — don't open ``progress_to_thread(...)`` around the
  workflow body. Default router sees ``None`` and is a no-op.
* **layer additional observers** — ``@receiver(brainstorm_progress)``
  on your own handler. Fans out alongside the default.
* **replace the default chat router** — disconnect + connect your
  own (see the divergent_convergent events.py module docstring for
  the same pattern's full recipe).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ballast.events import (
    Signal,
    chat_message_requested,
    progress_thread_var,
)


class BrainstormChose(BaseModel):
    """Divergent-convergent picked one ``TodoIdea``; HITL is about to open."""
    type: Literal["brainstorm-chose"] = "brainstorm-chose"
    title: str


class BrainstormSaved(BaseModel):
    """User approved the proposal; a Note was persisted.

    ``modified`` distinguishes plain-approve from modify-and-save —
    UIs can render slightly differently (e.g. "✓ saved" vs
    "✎ saved with edits")."""
    type: Literal["brainstorm-saved"] = "brainstorm-saved"
    title: str
    modified: bool = False


class BrainstormCancelled(BaseModel):
    """User rejected the proposal; no note saved."""
    type: Literal["brainstorm-cancelled"] = "brainstorm-cancelled"
    reason: str | None = None


class BrainstormTimedOut(BaseModel):
    """HITL timeout fired before the user responded; no note saved."""
    type: Literal["brainstorm-timed-out"] = "brainstorm-timed-out"


BrainstormEvent = (
    BrainstormChose
    | BrainstormSaved
    | BrainstormCancelled
    | BrainstormTimedOut
)
"""Discriminated union of every event the brainstorm workflow emits."""


brainstorm_progress: Signal = Signal("brainstorm.progress")
"""Module-level signal carrying each :data:`BrainstormEvent`.

Handlers receive ``(sender=None, event=...)``. The default chat
router (auto-connected below) is one such handler; apps can add
their own freely or replace the default."""


# ── Default chat router ────────────────────────────────────────────────


def format_for_chat(event: BrainstormEvent) -> str:
    """Render one event as a human-readable chat line.

    Override by monkey-patching this function at startup, or by
    replacing :func:`default_chat_router` entirely (disconnect +
    reconnect — see module docstring)."""
    if isinstance(event, BrainstormChose):
        return f'Chose: "{event.title}". Opening approval thread…'
    if isinstance(event, BrainstormSaved):
        suffix = " (with your edits)" if event.modified else ""
        return f'Saved your todo titled "{event.title}"{suffix}.'
    if isinstance(event, BrainstormCancelled):
        reason = (event.reason or "").strip()
        tail = f" ({reason})" if reason else ""
        return f"Todo creation was cancelled{tail}."
    if isinstance(event, BrainstormTimedOut):
        return "Todo approval timed out — nothing was saved."
    return ""


async def default_chat_router(
    sender: Any,
    *,
    event: BrainstormEvent,
    **_: Any,
) -> None:
    """Bundled :data:`brainstorm_progress` handler.

    Reads :data:`progress_thread_var` from the active context — if
    the workflow body didn't open a ``progress_to_thread(...)`` scope
    this is a no-op. If set, formats via :func:`format_for_chat` and
    publishes through :data:`chat_message_requested`.

    Auto-connected at module import."""
    thread_id = progress_thread_var.get()
    if thread_id is None:
        return
    text = format_for_chat(event)
    if not text:
        return
    await chat_message_requested.send(
        sender=sender, thread_id=thread_id, text=text,
    )


brainstorm_progress.connect(default_chat_router)


__all__ = [
    "BrainstormCancelled",
    "BrainstormChose",
    "BrainstormEvent",
    "BrainstormSaved",
    "BrainstormTimedOut",
    "brainstorm_progress",
    "default_chat_router",
    "format_for_chat",
]

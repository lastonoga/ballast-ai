"""Pattern-specific progress events for ``DivergentConvergent``.

Typed event vocabulary the pattern emits at every observable boundary
(branch enqueued / completed / failed, dedup completed, verify
completed, converge started / completed). Discriminated by ``type``
so handlers can ``match`` cleanly.

## Subscribing

Three flavours, in increasing order of customisation:

1. **Default chat narration** — already auto-connected. Wrap your
   pattern call in ``progress_to_thread(thread_id)`` and the bundled
   :func:`default_chat_router` posts one assistant message per event
   into that thread via :data:`chat_message_requested`::

       from ballast.events import progress_to_thread

       with progress_to_thread(parent_thread_id):
           chosen = await _divergent.run(topic)

   Skip the ``with`` block and the default handler is still connected
   but reads ``None`` from the contextvar → no-op. So opt-out is just
   "don't use the context manager".

2. **Additional subscribers** — connect anything alongside. Standard
   signal fan-out, both fire::

       from ballast.events import receiver

       @receiver(divergent_convergent_progress)
       async def to_metrics(sender, *, event, **_):
           if isinstance(event, BranchFailed):
               failed_counter.labels(label=event.label).inc()

3. **Replace the default** — disconnect :func:`default_chat_router`
   and connect your own with custom format / destination / filtering::

       from ballast.patterns.divergent_convergent.events import (
           default_chat_router, divergent_convergent_progress,
       )
       divergent_convergent_progress.disconnect(default_chat_router)

       @receiver(divergent_convergent_progress)
       async def my_router(sender, *, event, **_):
           # custom routing — Slack, your own thread, formatting, …
           ...

Handlers are dispatched in registration order; raised exceptions
abort the ``signal.send`` and propagate up through the pattern
body. The default chat router is async + safe to override.

The signal carries one kwarg, ``event``, whose type is the
:data:`DivergentEvent` union — handlers should ``isinstance``-dispatch
or ``match`` on ``event.type``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ballast.events import Signal, progress_thread_var


class BranchEnqueued(BaseModel):
    """A divergent branch has been placed on the queue, not yet started."""
    type: Literal["branch-enqueued"] = "branch-enqueued"
    label: str
    sample_idx: int


class BranchCompleted(BaseModel):
    """A divergent branch produced its hypothesis pool successfully."""
    type: Literal["branch-completed"] = "branch-completed"
    label: str
    sample_idx: int
    pool_size: int


class BranchFailed(BaseModel):
    """A divergent branch raised an exception."""
    type: Literal["branch-failed"] = "branch-failed"
    label: str
    sample_idx: int
    error_type: str


class DedupCompleted(BaseModel):
    """Dedup pass finished — fires only when a ``deduper`` is wired."""
    type: Literal["dedup-completed"] = "dedup-completed"
    input_count: int
    output_count: int


class VerifyCompleted(BaseModel):
    """Verifier pass finished — fires only when a ``verifier`` is wired."""
    type: Literal["verify-completed"] = "verify-completed"
    scored_count: int
    top_k_applied: int | None


class ConvergeStarted(BaseModel):
    """Synthesizer is about to run on the surviving candidate pool."""
    type: Literal["converge-started"] = "converge-started"
    candidate_count: int


class ConvergeCompleted(BaseModel):
    """Synthesizer returned its chosen output. Workflow returns next."""
    type: Literal["converge-completed"] = "converge-completed"


DivergentEvent = (
    BranchEnqueued
    | BranchCompleted
    | BranchFailed
    | DedupCompleted
    | VerifyCompleted
    | ConvergeStarted
    | ConvergeCompleted
)
"""Discriminated union of every event ``DivergentConvergent.run`` may
emit. Future kinds will extend this union; handlers should keep a
fall-through ``case _`` arm for forward compatibility."""


divergent_convergent_progress: Signal = Signal(
    "divergent_convergent.progress",
)
"""Module-level signal carrying each :data:`DivergentEvent` the pattern
emits. Handlers receive ``(sender=pattern_instance, event=...)``.

One signal per pattern (not per event type) keeps subscription light:
handlers ``isinstance``-dispatch internally and apps can filter to the
subset they care about.
"""


# ── Default chat router ────────────────────────────────────────────────


# Events the default router emits as typed UI parts. Apps subscribing
# to additional event types for non-UI side effects (metrics, audit)
# are independent — those handlers run regardless of this set.
_CHAT_RENDERED_EVENTS: tuple[type[BaseModel], ...] = (
    BranchEnqueued,
    BranchCompleted,
    BranchFailed,
    DedupCompleted,
    ConvergeStarted,
    ConvergeCompleted,
)


def _stable_message_id(event: DivergentEvent) -> str | None:
    """Compute the chat ``message_id`` for an event — reused by enqueued /
    completed / failed phases of the SAME branch so the row mutates
    rather than stacks.

    Scoped to the current DBOS workflow id so two concurrent brainstorm
    runs on the same thread don't collide on ``"branch:practical:0"``.

    Returns ``None`` for events whose row should be a fresh append
    instead of an in-place update.
    """
    from dbos import DBOS  # noqa: PLC0415

    try:
        scope = DBOS.workflow_id or "no-workflow"
    except Exception:  # noqa: BLE001 — DBOS not in workflow context
        scope = "no-workflow"

    if isinstance(event, BranchEnqueued | BranchCompleted | BranchFailed):
        return f"{scope}:branch:{event.label}:{event.sample_idx}"
    if isinstance(event, DedupCompleted):
        return f"{scope}:dedup"
    if isinstance(event, ConvergeStarted | ConvergeCompleted):
        return f"{scope}:converge"
    return None


async def default_chat_router(
    sender: Any,  # noqa: ARG001
    *,
    event: DivergentEvent,
    **_: Any,
) -> None:
    """Bundled :data:`divergent_convergent_progress` handler.

    Reads :data:`progress_thread_var` from the active context — if the
    app didn't open a ``progress_to_thread(...)`` scope it's a no-op.
    Otherwise writes a ``data-<event-type>`` part to the destination
    thread via the engine's :class:`ThreadEventBroadcaster`. That call
    persists the part, appends a ``message-added`` row to the event
    log, and publishes an ``EventNotification`` — all in one trip.

    Successive events for the SAME logical row (branch, dedup,
    converge) share a stable ``message_id`` (see
    :func:`_stable_message_id`), so the broadcaster upserts in place —
    a "branch enqueued" spinner morphs into a "branch completed"
    check on the same chat line instead of stacking up.

    ``VerifyCompleted`` is absorbed — it doesn't add UX value beyond
    what the surrounding workflow's own narration already says. The
    typed signal still fires for it so observers (analytics, audit)
    see it.

    Auto-connected at module import. Apps that want to replace its
    behaviour entirely should::

        divergent_convergent_progress.disconnect(default_chat_router)
        @receiver(divergent_convergent_progress)
        async def my_router(sender, *, event, **_): ...
    """
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    thread_id = progress_thread_var.get()
    if thread_id is None:
        return
    if not isinstance(event, _CHAT_RENDERED_EVENTS):
        return
    await get_ballast().broadcaster.emit_raw(
        thread_id,
        part={
            "type": f"data-{event.type}",
            "data": event.model_dump(mode="json"),
        },
        message_id=_stable_message_id(event),
        persistent=True,
    )


divergent_convergent_progress.connect(default_chat_router)


__all__ = [
    "BranchCompleted",
    "BranchEnqueued",
    "BranchFailed",
    "ConvergeCompleted",
    "ConvergeStarted",
    "DedupCompleted",
    "DivergentEvent",
    "VerifyCompleted",
    "default_chat_router",
    "divergent_convergent_progress",
]

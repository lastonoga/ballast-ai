"""Pattern-specific progress events for ``DivergentConvergent``.

Typed event vocabulary the pattern emits at every observable boundary
(branch enqueued / completed / failed, dedup completed, verify
completed, converge started / completed). Discriminated by ``type``
so handlers can ``match`` cleanly.

## How to subscribe

Two paths:

1. **Adapter helpers** (typical) — :mod:`ballast.events.adapters`
   exposes ``route_to_thread_as_text`` / ``route_to_thread_as_data``
   that connect a small handler to :data:`divergent_convergent_progress`
   and emit one chat message per event. App opts in inside its
   workflow body::

       from ballast.events.adapters import route_to_thread_as_text
       from ballast.patterns.divergent_convergent.events import (
           divergent_convergent_progress,
       )

       disconnect = route_to_thread_as_text(
           divergent_convergent_progress, thread_id=parent_id,
       )
       try:
           chosen = await _divergent.run(topic)
       finally:
           disconnect()

2. **Raw signal handler** — for non-thread destinations (Slack,
   metrics, audit log). Connect any sync/async receiver via the
   Django-style API::

       from ballast.events import receiver

       @receiver(divergent_convergent_progress)
       async def to_metrics(sender, *, event, **_):
           if isinstance(event, BranchFailed):
               failed_counter.labels(label=event.label).inc()

Handlers are dispatched in registration order; raised exceptions
abort the ``signal.send`` and propagate up through the pattern
body. Use :meth:`Signal.send_robust` semantics (or guard your
handler) if you want fail-quiet routing.

The signal carries one kwarg, ``event``, whose type is the
:data:`DivergentEvent` union — handlers should ``isinstance``-dispatch
or ``match`` on ``event.type``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ballast.events import Signal


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


__all__ = [
    "BranchCompleted",
    "BranchEnqueued",
    "BranchFailed",
    "ConvergeCompleted",
    "ConvergeStarted",
    "DedupCompleted",
    "DivergentEvent",
    "VerifyCompleted",
    "divergent_convergent_progress",
]

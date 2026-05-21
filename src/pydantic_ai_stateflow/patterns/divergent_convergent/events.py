"""Pattern-specific progress events emitted by ``DivergentConvergent.run``.

These are framework-level **typed** events. The pattern doesn't know
or care how they're rendered — it just produces them. Apps wire a
``on_progress`` callback to ``DivergentConvergent.run(...)`` that
maps the framework events to whatever delivery they like (thread
events, logfire spans, websocket pushes, …).

Discriminated by ``type`` so callers can ``match`` cleanly::

    async def on_progress(ev: DivergentEvent) -> None:
        match ev:
            case BranchCompleted(label=label, pool_size=n):
                ...
            case ConvergeStarted(candidate_count=n):
                ...

## Determinism / replay

The pattern's ``run`` body is a ``@Durable.workflow``. On crash
recovery the body replays in deterministic order:

- ``await handle.get_result()`` returns the memoised pool from the
  original execution, so ``BranchCompleted(pool_size=...)`` carries
  the same ``pool_size`` on replay.
- ``on_progress`` is invoked from inside the workflow body, so it
  too re-fires on replay. App-side mappings should be idempotent
  (e.g. ``ThreadEventStream`` reuses a stable ``message_id`` so
  upsert collapses retries).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel


class BranchEnqueued(BaseModel):
    """A divergent branch has been placed on the queue, not yet started.

    Fires once per (branch, sample) pair immediately before the
    enqueue call. The UI typically shows this as a ``running``
    state until ``BranchCompleted`` / ``BranchFailed`` arrives."""
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
    """A divergent branch raised an exception.

    With ``per_branch_failure="skip"`` (default) the workflow
    continues and this event tells the UI to mark that branch
    failed but keep going. With ``per_branch_failure="strict"``
    the workflow aborts immediately AFTER firing this event."""
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
emit. Future event kinds extend this — pattern matchers should
include a ``case _`` arm for forward compatibility."""

ProgressCallback = Callable[[DivergentEvent], Awaitable[None]]
"""Type alias for the optional callback passed to ``run(..., on_progress=...)``.

The callback runs **inside the workflow fiber** — exceptions raised
inside it are caught by the pattern and logged, NOT propagated, so a
broken UI mapping can't kill an in-flight brainstorm."""


__all__ = [
    "BranchCompleted",
    "BranchEnqueued",
    "BranchFailed",
    "ConvergeCompleted",
    "ConvergeStarted",
    "DedupCompleted",
    "DivergentEvent",
    "ProgressCallback",
    "VerifyCompleted",
]

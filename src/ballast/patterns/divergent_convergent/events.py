"""Pattern-specific progress events for ``DivergentConvergent``.

Typed event vocabulary that the pattern WILL emit at every observable
boundary (branch enqueued / completed / failed, dedup completed,
verify completed, converge started / completed). Discriminated by
``type`` so callers can ``match`` cleanly.

Status: the event types are defined but ``DivergentConvergent.run``
does NOT emit them yet. The previous ``on_progress=callable`` API
was removed because callable args can't cross a durable-workflow
boundary cleanly (local closures aren't picklable, and even with
module-level partials DBOS's serialization story is fragile).

## Planned re-introduction

The pattern will emit these typed events on the engine's
thread-event broadcaster (already wired by ``EventsProvider`` and
reached via ``get_engine().broadcaster``). Each event has a stable
wire name (``branch-enqueued``, ``branch-completed``, etc.) so apps
subscribe by name from outside the workflow body — no callback or
closure crosses the fiber boundary, recovery semantics stay clean,
and UIs map events to their own thread-event types as they please.

Apps that need per-branch live progress today should layer their
own emission inside their custom ``DivergentAgent`` implementations
(``.diverge`` runs in the branch fiber and CAN call into the
broadcaster directly).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
"""Discriminated union of every event ``DivergentConvergent.run`` will
emit once the broadcaster wiring lands."""


__all__ = [
    "BranchCompleted",
    "BranchEnqueued",
    "BranchFailed",
    "ConvergeCompleted",
    "ConvergeStarted",
    "DedupCompleted",
    "DivergentEvent",
    "VerifyCompleted",
]

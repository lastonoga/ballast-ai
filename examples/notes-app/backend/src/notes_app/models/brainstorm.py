"""Brainstorm-flow input/output envelopes."""

from uuid import UUID

from pydantic import BaseModel


class BrainstormTask(BaseModel):
    """Input to ``BrainstormFlow.run`` — one pydantic envelope so the
    workflow's call signature stays stable as the inputs grow (extra
    knobs like ``best_of_n_override``, ``locale`` etc. can be added
    without breaking callers)."""
    topic: str
    parent_thread_id: UUID


class BrainstormOutcome(BaseModel):
    """Output of ``BrainstormFlow.run``.

    The flow is fire-and-forget w.r.t. HITL: ``run`` returns AFTER the
    approval thread is opened but BEFORE the user approves/rejects.
    ``helper_thread_id`` is what the UI needs to scroll the sidebar
    to. ``proposed_title`` / ``proposed_body`` are included so
    observability (and any caller that wants to log what was
    proposed) doesn't need to peek into the helper thread."""
    helper_thread_id: UUID
    proposed_title: str
    proposed_body: str


__all__ = ["BrainstormOutcome", "BrainstormTask"]

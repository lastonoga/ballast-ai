"""Brainstorm-flow input/output envelopes."""

from uuid import UUID

from pydantic import BaseModel


class BrainstormTask(BaseModel):
    """Input to ``brainstorm`` — one pydantic envelope so the
    workflow's call signature stays stable as the inputs grow (extra
    knobs like ``best_of_n_override``, ``locale`` etc. can be added
    without breaking callers)."""
    topic: str
    parent_thread_id: UUID


class BrainstormOutcome(BaseModel):
    """Output of ``brainstorm``.

    The flow now runs end-to-end (diverge → ask → save), so the
    outcome describes the WHOLE run: what was proposed plus what
    actually got saved (``saved_title`` / ``saved_body`` are ``None``
    on reject / timeout). Observability + tests consume this; the
    HTTP route returns the workflow id immediately and the response
    body isn't streamed to the browser."""
    proposed_title: str
    proposed_body: str
    saved_title: str | None = None
    saved_body: str | None = None


__all__ = ["BrainstormOutcome", "BrainstormTask"]

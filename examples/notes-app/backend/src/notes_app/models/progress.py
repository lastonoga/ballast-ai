"""Live-progress thread-event payloads for the brainstorm flow."""

from typing import Literal, Optional

from pydantic import BaseModel


class BrainstormProgress(BaseModel):
    """Snapshot of where ``BrainstormFlow.run`` currently is.

    Frontend renders this as a single line that mutates: an icon
    flips ``running → ok`` per phase, with optional context in
    ``detail`` (e.g. the chosen idea's title once converge finishes).
    """
    step: Literal["diverge", "converge", "hitl"]
    status: Literal["running", "ok", "failed"]
    detail: Optional[str] = None


class BrainstormBranchProgress(BaseModel):
    """Per-branch live status during the divergent fan-out.

    One of these mutates per ``(label, sample_idx)`` pair as the
    branch transitions ``running → ok|failed``. Frontend renders the
    bundle so the user sees individual proposers tick off in parallel
    rather than a single opaque "brainstorming" spinner.
    """
    label: str
    sample_idx: int
    status: Literal["running", "ok", "failed"]
    pool_size: Optional[int] = None
    error_type: Optional[str] = None


__all__ = ["BrainstormBranchProgress", "BrainstormProgress"]

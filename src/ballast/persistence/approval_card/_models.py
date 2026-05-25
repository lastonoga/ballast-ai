"""``ApprovalCard`` — one human approval request awaiting a decision."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

CardStatus = Literal["pending", "approved", "rejected", "timeout"]


class ApprovalCard(BaseModel):
    """One pending / resolved approval request displayed in the inbox.

    ``id`` doubles as the HITL ``request_id`` so the wire topic
    (`f"hitl:{id}"`) is stable across the channel ↔ workflow ↔ router
    hops. ``payload`` is the channel's input model as JSON; ``resolution``
    is the verdict dump once decided.
    """

    id: str
    workflow_id: str
    respond_topic: str
    kind: str
    payload: dict[str, Any]
    parent_thread_id: str | None
    user_id: str | None
    status: CardStatus
    resolution: dict[str, Any] | None = None
    created_at: datetime
    resolved_at: datetime | None = None


__all__ = ["ApprovalCard", "CardStatus"]

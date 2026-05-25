"""Approval card REST + SSE endpoints.

  GET    /approvals                          → list pending (filtered by current_user_id)
  GET    /approvals/{card_id}                → single card (403 if not yours)
  POST   /approvals/{card_id}/decision       → verdict → Durable.send_async to the suspended workflow
  GET    /approvals/stream                   → SSE multiplexer (Task 10)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, TypeAdapter

from ballast.auth.context import current_user_id
from ballast.durable import Durable
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    approval_card_decided,
    card_kind_registry,
)
from ballast.persistence.approval_card import ApprovalCard

approvals_router = APIRouter(prefix="/approvals", tags=["approvals"])


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    modified: dict[str, Any] | None = None
    feedback:  str | None = None


@approvals_router.get("", response_model=list[ApprovalCard])
async def list_approvals(
    status: Literal["pending"] = Query("pending"),
    limit:  int = Query(50, ge=1, le=200),
) -> list[ApprovalCard]:
    """Pending approvals visible to the caller (filtered by user_id
    when ``current_user_id()`` is set, unscoped otherwise)."""
    from ballast.persistence.approval_card import approval_card_repo  # noqa: PLC0415
    return await approval_card_repo.list_pending(limit=limit)


@approvals_router.get("/{card_id}", response_model=ApprovalCard)
async def get_approval(card_id: str) -> ApprovalCard:
    from ballast.persistence.approval_card import approval_card_repo  # noqa: PLC0415
    card = await approval_card_repo.get(card_id)
    if card is None:
        raise HTTPException(404, "Approval not found")
    return card


@approvals_router.post("/{card_id}/decision", response_model=ApprovalCard)
async def decide_approval(
    card_id: str, body: DecisionRequest,
) -> ApprovalCard:
    from ballast.persistence.approval_card import approval_card_repo  # noqa: PLC0415

    card = await approval_card_repo.get(card_id)
    if card is None:
        raise HTTPException(404, "Approval not found")
    if current_user_id() is not None and card.user_id != current_user_id():
        raise HTTPException(403, "Not your approval")
    if card.status != "pending":
        raise HTTPException(
            409, f"Card already {card.status}",
        )

    # Validate ``modified`` (if present) against the kind's registered model.
    payload_model = card_kind_registry.get(card.kind)
    if body.modified is not None and payload_model is not None:
        modified_typed = TypeAdapter(payload_model).validate_python(body.modified)
    else:
        modified_typed = body.modified

    verdict = CardVerdict[payload_model](  # type: ignore[valid-type]
        decision=body.decision,
        modified=modified_typed,
        feedback=body.feedback,
        answered_at=datetime.now(timezone.utc),
    ) if payload_model is not None else CardVerdict(
        decision=body.decision,
        modified=modified_typed,
        feedback=body.feedback,
        answered_at=datetime.now(timezone.utc),
    )

    await Durable.send_async(
        destination_id=card.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=card.respond_topic,
    )
    resolved = await approval_card_repo.resolve(card.id, verdict=verdict)
    await approval_card_decided.send(None, card=resolved)
    return resolved

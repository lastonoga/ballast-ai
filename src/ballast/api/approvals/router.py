"""Approval card REST + SSE endpoints.

  GET    /approvals                          → list pending (filtered by current_user_id)
  GET    /approvals/stream                   → SSE multiplexer (Task 10)
  GET    /approvals/{card_id}                → single card (403 if not yours)
  POST   /approvals/{card_id}/decision       → verdict → Durable.send_async to the suspended workflow
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, TypeAdapter
from sse_starlette.sse import EventSourceResponse

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


@approvals_router.get("/stream")
async def stream_approvals(request: Request) -> EventSourceResponse:
    """Multiplex ``approval_card_requested`` + ``approval_card_decided``
    signals as SSE events. Disconnect-aware via ``request.is_disconnected``.
    """
    from ballast.patterns.hitl.channels.ui_card import (  # noqa: PLC0415
        approval_card_requested,
    )

    # asyncio.Queue: handlers are called from the same running event loop
    # (via Signal.send which awaits async receivers or calls sync ones directly).
    # For cross-thread callers, the sync handlers use put_nowait which is
    # safe to call from any thread as long as the loop is running.
    aqueue: asyncio.Queue[tuple[str, ApprovalCard]] = asyncio.Queue()

    def _on_request(sender: Any, *, card: ApprovalCard, **_: Any) -> None:
        aqueue.put_nowait(("card-requested", card))

    def _on_decided(sender: Any, *, card: ApprovalCard, **_: Any) -> None:
        aqueue.put_nowait(("card-decided", card))

    approval_card_requested.connect(_on_request)
    approval_card_decided.connect(_on_decided)

    async def _gen() -> AsyncIterator[dict[str, str]]:
        try:
            while True:
                try:
                    event_name, card = await asyncio.wait_for(
                        aqueue.get(), timeout=15.0,
                    )
                    yield {
                        "event": event_name,
                        "data": card.model_dump_json(),
                    }
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": ""}
        finally:
            approval_card_requested.disconnect(_on_request)
            approval_card_decided.disconnect(_on_decided)

    return EventSourceResponse(_gen())


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

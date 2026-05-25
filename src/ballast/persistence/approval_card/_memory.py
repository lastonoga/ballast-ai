"""In-memory ``ApprovalCardRepository`` — tests + local dev."""
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from ballast.auth.context import current_user_id
from ballast.persistence.approval_card._models import ApprovalCard
from ballast.persistence.approval_card._repo import ApprovalCardRepository


class InMemoryApprovalCardRepository(ApprovalCardRepository):
    """Process-local dict-backed repo. Filters by ``current_user_id``
    when set; returns all rows otherwise (admin / single-user mode)."""

    def __init__(self) -> None:
        self._rows: dict[str, ApprovalCard] = {}

    async def add(self, card: ApprovalCard) -> None:
        self._rows[card.id] = card

    async def get(self, card_id: str) -> ApprovalCard | None:
        card = self._rows.get(card_id)
        if card is None:
            return None
        scope = current_user_id()
        if scope is not None and card.user_id != scope:
            return None
        return card

    async def list_pending(
        self, *, limit: int = 50,
    ) -> list[ApprovalCard]:
        scope = current_user_id()
        out = [
            c for c in self._rows.values()
            if c.status == "pending"
            and (scope is None or c.user_id == scope)
        ]
        out.sort(key=lambda c: c.created_at)
        return out[:limit]

    async def resolve(
        self, card_id: str, *, verdict: BaseModel,
    ) -> ApprovalCard:
        card = self._rows.get(card_id)
        if card is None:
            raise KeyError(card_id)
        decision = getattr(verdict, "decision", None)
        match decision:
            case "approve": new_status = "approved"
            case "reject":  new_status = "rejected"
            case _:         new_status = "timeout"
        updated = card.model_copy(update={
            "status": new_status,
            "resolution": verdict.model_dump(mode="json"),
            "resolved_at": datetime.now(UTC),
        })
        self._rows[card_id] = updated
        return updated


__all__ = ["InMemoryApprovalCardRepository"]

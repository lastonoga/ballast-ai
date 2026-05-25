"""Postgres-backed ``ApprovalCardRepository`` — mirrors SqlThreadRepository pattern."""
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel
from sqlmodel import select

from ballast.auth.context import current_user_id
from ballast.persistence._sql_base import SqlSessionMixin
from ballast.persistence.approval_card._models import ApprovalCard
from ballast.persistence.approval_card._repo import ApprovalCardRepository


class SqlApprovalCardRepository(SqlSessionMixin, ApprovalCardRepository):
    """Postgres-backed approval-card repo.

    Visibility (``list_pending`` / ``get``) filters by
    ``current_user_id()`` when set, otherwise returns all rows
    (admin / single-user mode). Mirrors the in-memory impl.
    """

    async def add(self, card: ApprovalCard) -> None:
        async with self._tx() as session:
            session.add(card)

    async def get(self, card_id: str) -> ApprovalCard | None:
        async with self._session() as session:
            card = await session.get(ApprovalCard, card_id)
            if card is None:
                return None
            scope = current_user_id()
            if scope is not None and card.user_id != scope:
                return None
            return card

    async def list_pending(self, *, limit: int = 50) -> list[ApprovalCard]:
        scope = current_user_id()
        async with self._session() as session:
            stmt = select(ApprovalCard).where(ApprovalCard.status == "pending")
            if scope is not None:
                stmt = stmt.where(ApprovalCard.user_id == scope)
            stmt = stmt.order_by(ApprovalCard.created_at).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def resolve(
        self,
        card_id: str,
        *,
        verdict: BaseModel,
    ) -> ApprovalCard:
        async with self._tx() as session:
            card = await session.get(ApprovalCard, card_id)
            if card is None:
                raise KeyError(card_id)
            decision = getattr(verdict, "decision", None)
            match decision:
                case "approve":
                    card.status = "approved"
                case "reject":
                    card.status = "rejected"
                case _:
                    card.status = "timeout"
            card.resolution = verdict.model_dump(mode="json")
            card.resolved_at = datetime.now(UTC)
            session.add(card)
            await session.flush()
            await session.refresh(card)
            return card


__all__ = ["SqlApprovalCardRepository"]

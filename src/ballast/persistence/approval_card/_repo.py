"""``ApprovalCardRepository`` Protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ballast.persistence.approval_card._models import ApprovalCard


@runtime_checkable
class ApprovalCardRepository(Protocol):
    """Read/write of approval card rows.

    Visibility (``list_pending`` / ``get``) is the implementation's
    concern: the in-memory and SQL repos filter by ``current_user_id()``
    when set, returning all rows when unscoped.
    """

    async def add(self, card: ApprovalCard) -> None: ...

    async def get(self, card_id: str) -> ApprovalCard | None: ...

    async def list_pending(
        self, *, limit: int = 50,
    ) -> list[ApprovalCard]: ...

    async def resolve(
        self, card_id: str, *, verdict: BaseModel,
    ) -> ApprovalCard: ...


__all__ = ["ApprovalCardRepository"]

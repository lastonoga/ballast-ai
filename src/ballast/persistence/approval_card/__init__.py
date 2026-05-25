"""Approval card persistence — model + Protocol + in-memory impl."""
from ballast.persistence.approval_card._memory import (
    InMemoryApprovalCardRepository,
)
from ballast.persistence.approval_card._models import (
    ApprovalCard,
    CardStatus,
)
from ballast.persistence.approval_card._repo import ApprovalCardRepository

__all__ = [
    "ApprovalCard",
    "ApprovalCardRepository",
    "CardStatus",
    "InMemoryApprovalCardRepository",
    "approval_card_repo",
]

# Module-level singleton, reassigned at app-build time when the user
# configures a custom repo (see ``Ballast.with_approval_repo``). Tests
# monkeypatch this attribute directly — same pattern as ``notes_repo``.
approval_card_repo: ApprovalCardRepository = InMemoryApprovalCardRepository()

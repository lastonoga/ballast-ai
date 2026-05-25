"""The module exposes a swappable ``approval_card_repo`` singleton —
production reassigns via ``Ballast.with_approval_repo(...)``; tests
monkeypatch the same attribute.
"""
from __future__ import annotations

import pytest

from ballast.persistence import approval_card as mod


def test_default_singleton_is_inmemory() -> None:
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    assert isinstance(mod.approval_card_repo, InMemoryApprovalCardRepository)


@pytest.mark.asyncio
async def test_singleton_is_monkeypatchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests swap the singleton via monkeypatch — same convention as
    notes_repo."""
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    fresh = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo",
        fresh,
    )
    assert mod.approval_card_repo is fresh

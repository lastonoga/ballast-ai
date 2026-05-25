"""``Ballast.with_approval_repo`` swaps the module singleton at build."""
from __future__ import annotations

import ballast
from ballast.persistence.approval_card import (
    InMemoryApprovalCardRepository,
)
from ballast.settings import BallastSettings


def test_with_approval_repo_installs_singleton() -> None:
    fresh = InMemoryApprovalCardRepository()
    b = ballast.Ballast(BallastSettings()).with_approval_repo(fresh)

    from ballast.persistence import approval_card as mod
    assert mod.approval_card_repo is fresh

    b  # noqa: B018 — touch the var

"""Integration tests for SqlApprovalCardRepository against a real Postgres DB."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from ballast.auth.context import acting_as
from ballast.persistence.approval_card import (
    ApprovalCard,
    SqlApprovalCardRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _card(id_: str, *, user_id: str | None, status: str = "pending") -> ApprovalCard:
    return ApprovalCard(
        id=id_,
        workflow_id=f"wf-{id_}",
        respond_topic=f"hitl:{id_}",
        kind="note.create",
        payload={},
        parent_thread_id=None,
        user_id=user_id,
        status=status,  # type: ignore[arg-type]
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_add_then_get(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Add a card, then retrieve it by id."""
    repo = SqlApprovalCardRepository(session_factory)
    card = _card("sql-a", user_id="u-1")
    await repo.add(card)

    got = await repo.get("sql-a")
    assert got is not None
    assert got.id == "sql-a"
    assert got.workflow_id == "wf-sql-a"
    assert got.status == "pending"


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = SqlApprovalCardRepository(session_factory)
    assert await repo.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_list_pending_unscoped_returns_all_pending(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """No acting_as scope → returns all pending rows (admin mode)."""
    repo = SqlApprovalCardRepository(session_factory)
    await repo.add(_card("sql-b1", user_id="u-1"))
    await repo.add(_card("sql-b2", user_id="u-2"))
    # approved card — must NOT appear in list_pending
    await repo.add(_card("sql-b3", user_id="u-1", status="approved"))

    listed = await repo.list_pending()
    ids = {c.id for c in listed}
    assert "sql-b1" in ids
    assert "sql-b2" in ids
    assert "sql-b3" not in ids


@pytest.mark.asyncio
async def test_list_pending_scoped_filters_by_user(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """acting_as scope → only cards for that user are returned."""
    repo = SqlApprovalCardRepository(session_factory)
    await repo.add(_card("sql-c1", user_id="u-10"))
    await repo.add(_card("sql-c2", user_id="u-20"))

    with acting_as("u-10"):
        listed = await repo.list_pending()

    ids = {c.id for c in listed}
    assert "sql-c1" in ids
    assert "sql-c2" not in ids


@pytest.mark.asyncio
async def test_get_scoped_hides_other_user_card(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """get() respects current_user_id scope — returns None for another user's card."""
    repo = SqlApprovalCardRepository(session_factory)
    await repo.add(_card("sql-d1", user_id="u-alpha"))

    with acting_as("u-beta"):
        got = await repo.get("sql-d1")
    assert got is None


@pytest.mark.asyncio
async def test_resolve_approve(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """resolve() with decision='approve' flips status + stamps resolution."""
    from pydantic import BaseModel  # noqa: PLC0415

    class _V(BaseModel):
        decision: str

    repo = SqlApprovalCardRepository(session_factory)
    await repo.add(_card("sql-e1", user_id="u-1"))

    resolved = await repo.resolve("sql-e1", verdict=_V(decision="approve"))
    assert resolved.status == "approved"
    assert resolved.resolution == {"decision": "approve"}
    assert resolved.resolved_at is not None

    # Verify the update landed in the DB.
    reloaded = await repo.get("sql-e1")
    assert reloaded is not None
    assert reloaded.status == "approved"


@pytest.mark.asyncio
async def test_resolve_reject(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    from pydantic import BaseModel  # noqa: PLC0415

    class _V(BaseModel):
        decision: str

    repo = SqlApprovalCardRepository(session_factory)
    await repo.add(_card("sql-f1", user_id="u-1"))

    resolved = await repo.resolve("sql-f1", verdict=_V(decision="reject"))
    assert resolved.status == "rejected"


@pytest.mark.asyncio
async def test_resolve_unknown_raises(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    from pydantic import BaseModel  # noqa: PLC0415

    class _V(BaseModel):
        decision: str

    repo = SqlApprovalCardRepository(session_factory)
    with pytest.raises(KeyError):
        await repo.resolve("no-such-card", verdict=_V(decision="approve"))

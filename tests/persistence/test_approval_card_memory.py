"""``InMemoryApprovalCardRepository`` — add / get / list_pending / resolve.

Per-user visibility is enforced at the repo edge by reading
``current_user_id()``. Tests exercise both the unscoped (None) and
scoped behaviors.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.auth.context import acting_as
from ballast.persistence.approval_card import (
    ApprovalCard,
    InMemoryApprovalCardRepository,
)


def _card(id_: str, *, user_id: str | None, status: str = "pending") -> ApprovalCard:
    return ApprovalCard(
        id=id_, workflow_id=f"wf-{id_}",
        respond_topic=f"hitl:{id_}", kind="note.create",
        payload={}, parent_thread_id=None, user_id=user_id,
        status=status,  # type: ignore[arg-type]
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_add_then_get() -> None:
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))
    got = await repo.get("a")
    assert got is not None and got.id == "a"


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown() -> None:
    repo = InMemoryApprovalCardRepository()
    assert await repo.get("nope") is None


@pytest.mark.asyncio
async def test_list_pending_filters_by_current_user_id() -> None:
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))
    await repo.add(_card("b", user_id="u-2"))
    await repo.add(_card("c", user_id="u-1", status="approved"))

    with acting_as("u-1"):
        listed = await repo.list_pending()
    ids = [c.id for c in listed]
    assert ids == ["a"]  # only pending + matches u-1


@pytest.mark.asyncio
async def test_list_pending_unscoped_returns_all_pending() -> None:
    """No acting_as scope → no user filter (admin / single-user use)."""
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))
    await repo.add(_card("b", user_id="u-2"))
    await repo.add(_card("c", user_id=None, status="approved"))

    listed = await repo.list_pending()
    assert {c.id for c in listed} == {"a", "b"}


@pytest.mark.asyncio
async def test_resolve_flips_status_and_stamps_resolution() -> None:
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))

    from pydantic import BaseModel
    class _V(BaseModel): decision: str

    resolved = await repo.resolve("a", verdict=_V(decision="approve"))
    assert resolved.status == "approved"
    assert resolved.resolution == {"decision": "approve"}
    assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_resolve_unknown_raises() -> None:
    repo = InMemoryApprovalCardRepository()
    from pydantic import BaseModel
    class _V(BaseModel): decision: str

    with pytest.raises(KeyError):
        await repo.resolve("nope", verdict=_V(decision="approve"))

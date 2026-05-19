"""F6 — Thread CRUD additions (list/rename/archive/unarchive/delete) on InMemory."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.thread import (
    InMemoryThreadRepository,
    ThreadPurpose,
    ThreadStatus,
)


@pytest.fixture
def tenant_id():
    return uuid4()


@pytest.fixture
def other_tenant_id():
    return uuid4()


@pytest.fixture
def repo() -> InMemoryThreadRepository:
    return InMemoryThreadRepository()


@pytest.mark.asyncio
async def test_list_returns_newest_first(repo, tenant_id):
    t1 = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value, purpose_metadata={},
        actor_id="a", tenant_id=tenant_id,
    )
    # Ensure distinct created_at by yielding to the loop.
    await asyncio.sleep(0.01)
    t2 = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value, purpose_metadata={},
        actor_id="a", tenant_id=tenant_id,
    )
    await asyncio.sleep(0.01)
    t3 = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value, purpose_metadata={},
        actor_id="a", tenant_id=tenant_id,
    )
    listed = await repo.list_(tenant_id=tenant_id)
    assert [t.id for t in listed] == [t3.id, t2.id, t1.id]


@pytest.mark.asyncio
async def test_list_excludes_archived_by_default(repo, tenant_id):
    t1 = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    t2 = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.archive(t1.id, tenant_id=tenant_id)
    listed = await repo.list_(tenant_id=tenant_id)
    ids = [t.id for t in listed]
    assert t1.id not in ids
    assert t2.id in ids


@pytest.mark.asyncio
async def test_list_includes_archived_when_flag_set(repo, tenant_id):
    t1 = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.archive(t1.id, tenant_id=tenant_id)
    listed = await repo.list_(tenant_id=tenant_id, include_archived=True)
    assert t1.id in [t.id for t in listed]


@pytest.mark.asyncio
async def test_list_isolated_by_tenant(repo, tenant_id, other_tenant_id):
    await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="b",
        tenant_id=other_tenant_id,
    )
    listed = await repo.list_(tenant_id=tenant_id)
    assert len(listed) == 1
    assert listed[0].tenant_id == tenant_id


# ── F18: offset pagination ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_offset_skips_first_n(repo, tenant_id):
    created = []
    for _ in range(5):
        t = await repo.create(
            purpose="conversation", purpose_metadata={}, actor_id="a",
            tenant_id=tenant_id,
        )
        created.append(t)
        # ensure distinct created_at so newest-first order is deterministic
        await asyncio.sleep(0.01)
    # Newest-first: created[4], created[3], created[2], created[1], created[0].
    # limit=2, offset=2 → created[2], created[1].
    page = await repo.list_(tenant_id=tenant_id, limit=2, offset=2)
    assert [t.id for t in page] == [created[2].id, created[1].id]


@pytest.mark.asyncio
async def test_list_offset_beyond_total_returns_empty(repo, tenant_id):
    for _ in range(3):
        await repo.create(
            purpose="conversation", purpose_metadata={}, actor_id="a",
            tenant_id=tenant_id,
        )
    page = await repo.list_(tenant_id=tenant_id, limit=10, offset=100)
    assert page == []


@pytest.mark.asyncio
async def test_rename_sets_title(repo, tenant_id):
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    renamed = await repo.rename(t.id, title="Trip planning", tenant_id=tenant_id)
    assert renamed.title == "Trip planning"
    loaded = await repo.load(t.id, tenant_id=tenant_id)
    assert loaded.title == "Trip planning"
    # null clears the title
    cleared = await repo.rename(t.id, title=None, tenant_id=tenant_id)
    assert cleared.title is None


@pytest.mark.asyncio
async def test_rename_404_cross_tenant(repo, tenant_id, other_tenant_id):
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    with pytest.raises(KeyError):
        await repo.rename(t.id, title="x", tenant_id=other_tenant_id)


@pytest.mark.asyncio
async def test_archive_sets_status(repo, tenant_id):
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    archived = await repo.archive(t.id, tenant_id=tenant_id)
    assert archived.status == ThreadStatus.ARCHIVED
    # add_message must still work on an ARCHIVED thread
    msg = await repo.add_message(
        t.id, role="user", parts=[{"type": "text", "text": "still here"}],
        tenant_id=tenant_id,
    )
    assert msg.role == "user"


@pytest.mark.asyncio
async def test_archive_404_cross_tenant(repo, tenant_id, other_tenant_id):
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    with pytest.raises(KeyError):
        await repo.archive(t.id, tenant_id=other_tenant_id)


@pytest.mark.asyncio
async def test_unarchive_restores_status(repo, tenant_id):
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.archive(t.id, tenant_id=tenant_id)
    restored = await repo.unarchive(t.id, tenant_id=tenant_id)
    assert restored.status == ThreadStatus.OPEN


@pytest.mark.asyncio
async def test_delete_is_idempotent(repo, tenant_id):
    # Deleting an unknown thread is a no-op (not an error).
    await repo.delete(uuid4(), tenant_id=tenant_id)
    # Deleting twice is a no-op.
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.delete(t.id, tenant_id=tenant_id)
    await repo.delete(t.id, tenant_id=tenant_id)
    assert await repo.load(t.id, tenant_id=tenant_id) is None


@pytest.mark.asyncio
async def test_delete_removes_messages_too(repo, tenant_id):
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.add_message(
        t.id, role="user", parts=[{"type": "text", "text": "hi"}],
        tenant_id=tenant_id,
    )
    await repo.delete(t.id, tenant_id=tenant_id)
    # history on a deleted thread returns empty
    assert await repo.history(t.id, tenant_id=tenant_id) == []


@pytest.mark.asyncio
async def test_delete_cross_tenant_is_noop(repo, tenant_id, other_tenant_id):
    t = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a",
        tenant_id=tenant_id,
    )
    # Cross-tenant delete must NOT remove the thread.
    await repo.delete(t.id, tenant_id=other_tenant_id)
    loaded = await repo.load(t.id, tenant_id=tenant_id)
    assert loaded is not None

"""F6 — Thread CRUD additions (list/update_metadata/archive/unarchive/delete) on InMemory."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from ballast.persistence.thread import (
    InMemoryThreadRepository,
    ThreadStatus,
)


@pytest.fixture
def repo() -> InMemoryThreadRepository:
    return InMemoryThreadRepository()


@pytest.mark.asyncio
async def test_list_returns_newest_first(repo):
    t1 = await repo.create(agent="conversation", metadata={})
    await asyncio.sleep(0.01)
    t2 = await repo.create(agent="conversation", metadata={})
    await asyncio.sleep(0.01)
    t3 = await repo.create(agent="conversation", metadata={})
    listed = await repo.list_()
    assert [t.id for t in listed] == [t3.id, t2.id, t1.id]


@pytest.mark.asyncio
async def test_list_excludes_archived_by_default(repo):
    t1 = await repo.create(agent="conversation", metadata={})
    t2 = await repo.create(agent="conversation", metadata={})
    await repo.archive(t1.id)
    listed = await repo.list_()
    ids = [t.id for t in listed]
    assert t1.id not in ids
    assert t2.id in ids


@pytest.mark.asyncio
async def test_list_includes_archived_when_flag_set(repo):
    t1 = await repo.create(agent="conversation", metadata={})
    await repo.archive(t1.id)
    listed = await repo.list_(include_archived=True)
    assert t1.id in [t.id for t in listed]


# F18: offset pagination


@pytest.mark.asyncio
async def test_list_offset_skips_first_n(repo):
    created = []
    for _ in range(5):
        t = await repo.create(agent="conversation", metadata={})
        created.append(t)
        await asyncio.sleep(0.01)
    # Newest-first; limit=2, offset=2 -> created[2], created[1].
    page = await repo.list_(limit=2, offset=2)
    assert [t.id for t in page] == [created[2].id, created[1].id]


@pytest.mark.asyncio
async def test_list_offset_beyond_total_returns_empty(repo):
    for _ in range(3):
        await repo.create(agent="conversation", metadata={})
    page = await repo.list_(limit=10, offset=100)
    assert page == []


@pytest.mark.asyncio
async def test_archive_sets_status(repo):
    t = await repo.create(agent="conversation", metadata={})
    archived = await repo.archive(t.id)
    assert archived.status == ThreadStatus.ARCHIVED
    # add_message must still work on an ARCHIVED thread
    msg = await repo.add_message(
        t.id, role="user", parts=[{"type": "text", "text": "still here"}],
    )
    assert msg.role == "user"


@pytest.mark.asyncio
async def test_unarchive_restores_status(repo):
    t = await repo.create(agent="conversation", metadata={})
    await repo.archive(t.id)
    restored = await repo.unarchive(t.id)
    assert restored.status == ThreadStatus.OPEN


@pytest.mark.asyncio
async def test_delete_is_idempotent(repo):
    # Deleting an unknown thread is a no-op (not an error).
    await repo.delete(uuid4())
    # Deleting twice is a no-op.
    t = await repo.create(agent="conversation", metadata={})
    await repo.delete(t.id)
    await repo.delete(t.id)
    assert await repo.load(t.id) is None


@pytest.mark.asyncio
async def test_delete_removes_messages_too(repo):
    t = await repo.create(agent="conversation", metadata={})
    await repo.add_message(
        t.id, role="user", parts=[{"type": "text", "text": "hi"}],
    )
    await repo.delete(t.id)
    assert await repo.history(t.id) == []

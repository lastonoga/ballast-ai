"""Integration tests for PostgresThreadRepository against a real Postgres DB."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow
from pydantic_ai_stateflow.persistence.thread import (
    PostgresThreadRepository,
    ThreadPurpose,
)

# ── helpers ──────────────────────────────────────────────────────────────────


async def _insert_tenant(factory: async_sessionmaker[AsyncSession]) -> TenantRow:
    """Insert a TenantRow and commit so FK constraints are satisfied."""
    async with factory() as session, session.begin():
        row = TenantRow(name=f"tenant-{uuid4().hex[:8]}")
        session.add(row)
    await session.close()
    return row


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_and_load_thread(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Create a thread inside a UoW, then reload it in a fresh session."""
    tenant = await _insert_tenant(session_factory)

    # Write: create thread
    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        thread = await repo.create(
            purpose=ThreadPurpose.CONVERSATION.value,
            purpose_metadata={"source": "test"},
            actor_id="user-42",
            tenant_id=tenant.id,
        )
        thread_id = thread.id

    # Read in a fresh session to verify persistence
    async with session_factory() as session:
        repo = PostgresThreadRepository(session)
        loaded = await repo.load(thread_id, tenant_id=tenant.id)

    assert loaded is not None
    assert loaded.id == thread_id
    assert loaded.actor_id == "user-42"
    assert loaded.tenant_id == tenant.id
    assert loaded.purpose_metadata == {"source": "test"}


@pytest.mark.asyncio
async def test_load_cross_tenant_returns_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Loading a thread with the wrong tenant_id must return None."""
    tenant_a = await _insert_tenant(session_factory)
    tenant_b = await _insert_tenant(session_factory)

    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        thread = await repo.create(
            purpose=ThreadPurpose.HITL.value,
            purpose_metadata={},
            actor_id="actor-1",
            tenant_id=tenant_a.id,
        )
        thread_id = thread.id

    async with session_factory() as session:
        repo = PostgresThreadRepository(session)
        result = await repo.load(thread_id, tenant_id=tenant_b.id)

    assert result is None


@pytest.mark.asyncio
async def test_add_message_and_history(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Add two messages and verify history returns them oldest-first."""
    tenant = await _insert_tenant(session_factory)

    # Create thread
    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        thread = await repo.create(
            purpose=ThreadPurpose.CONVERSATION.value,
            purpose_metadata={},
            actor_id="bot",
            tenant_id=tenant.id,
        )
        thread_id = thread.id

    # Add messages in separate transactions
    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        await repo.add_message(
            thread_id,
            role="user",
            parts=[{"kind": "text", "content": "hello"}],
            tenant_id=tenant.id,
        )

    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        await repo.add_message(
            thread_id,
            role="assistant",
            parts=[{"kind": "text", "content": "world"}],
            tenant_id=tenant.id,
        )

    # Read history
    async with session_factory() as session:
        repo = PostgresThreadRepository(session)
        history = await repo.history(thread_id, tenant_id=tenant.id, limit=10)

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].parts[0]["content"] == "hello"
    assert history[1].role == "assistant"
    assert history[1].parts[0]["content"] == "world"


@pytest.mark.asyncio
async def test_add_message_wrong_tenant_raises_key_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """add_message for wrong tenant must raise KeyError."""
    tenant_a = await _insert_tenant(session_factory)
    tenant_b = await _insert_tenant(session_factory)

    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        thread = await repo.create(
            purpose=ThreadPurpose.CONVERSATION.value,
            purpose_metadata={},
            actor_id="a",
            tenant_id=tenant_a.id,
        )
        thread_id = thread.id

    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        with pytest.raises(KeyError):
            await repo.add_message(
                thread_id,
                role="user",
                parts=[],
                tenant_id=tenant_b.id,
            )

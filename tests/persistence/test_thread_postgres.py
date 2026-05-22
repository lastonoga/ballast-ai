"""Integration tests for PostgresThreadRepository against a real Postgres DB."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ballast.persistence.thread import (
    PostgresThreadRepository,
)


@pytest.mark.asyncio
async def test_create_and_load_thread(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Create a thread inside a UoW, then reload it in a fresh session."""
    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        thread = await repo.create(
            agent="conversation",
            metadata={"source": "test"},
        )
        thread_id = thread.id

    async with session_factory() as session:
        repo = PostgresThreadRepository(session)
        loaded = await repo.load(thread_id)

    assert loaded is not None
    assert loaded.id == thread_id
    assert loaded.metadata_ == {"source": "test"}


@pytest.mark.asyncio
async def test_add_message_and_history(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Add two messages and verify history returns them oldest-first."""
    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        thread = await repo.create(
            agent="conversation",
            metadata={},
        )
        thread_id = thread.id

    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        await repo.add_message(
            thread_id,
            role="user",
            parts=[{"kind": "text", "content": "hello"}],
        )

    async with session_factory() as session, session.begin():
        repo = PostgresThreadRepository(session)
        await repo.add_message(
            thread_id,
            role="assistant",
            parts=[{"kind": "text", "content": "world"}],
        )

    async with session_factory() as session:
        repo = PostgresThreadRepository(session)
        history = await repo.history(thread_id, limit=10)

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].parts[0]["content"] == "hello"
    assert history[1].role == "assistant"
    assert history[1].parts[0]["content"] == "world"

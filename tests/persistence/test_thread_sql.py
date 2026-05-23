"""Integration tests for SqlThreadRepository against a real Postgres DB."""
# Note: fixtures provide a Postgres session_factory; SqlThreadRepository
# itself is backend-agnostic and is also exercised on sqlite via the
# notes-app integration tests.

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ballast.persistence.thread import (
    Message,
    SqlThreadRepository,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@pytest.mark.asyncio
async def test_create_and_load_thread(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Create a thread, then reload it (commit-per-method owned by repo)."""
    repo = SqlThreadRepository(session_factory)
    thread = await repo.create(agent="conversation", metadata={"source": "test"})

    loaded = await repo.load(thread.id)

    assert loaded is not None
    assert loaded.id == thread.id
    assert loaded.metadata_ == {"source": "test"}


@pytest.mark.asyncio
async def test_add_message_and_history(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Add two messages and verify history returns them oldest-first."""
    repo = SqlThreadRepository(session_factory)
    thread = await repo.create(agent="conversation", metadata={})

    await repo.add_message(
        thread.id,
        role="user",
        parts=[{"kind": "text", "content": "hello"}],
        silent=True,
    )
    await repo.add_message(
        thread.id,
        role="assistant",
        parts=[{"kind": "text", "content": "world"}],
        silent=True,
    )

    history = await repo.history(thread.id, limit=10)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].parts[0]["content"] == "hello"
    assert history[1].role == "assistant"
    assert history[1].parts[0]["content"] == "world"


@pytest.mark.asyncio
async def test_add_message_emits_message_added_signal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``add_message`` self-emits ``message_added`` after commit."""
    from ballast.events import message_added  # noqa: PLC0415

    repo = SqlThreadRepository(session_factory)
    thread = await repo.create(agent="conversation", metadata={})

    received: list[tuple[UUID, Message]] = []

    async def _receiver(sender: object, **kw: object) -> None:
        received.append((kw["thread_id"], kw["message"]))  # type: ignore[arg-type]

    message_added.connect(_receiver)
    try:
        msg = await repo.add_message(
            thread.id,
            role="user",
            parts=[{"kind": "text", "content": "hi"}],
        )
    finally:
        message_added.disconnect(_receiver)

    assert len(received) == 1
    assert received[0][0] == thread.id
    assert received[0][1].id == msg.id


@pytest.mark.asyncio
async def test_add_message_silent_skips_signal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``silent=True`` skips the post-commit signal but still persists."""
    from ballast.events import message_added  # noqa: PLC0415

    repo = SqlThreadRepository(session_factory)
    thread = await repo.create(agent="conversation", metadata={})

    received: list[object] = []

    async def _receiver(sender: object, **kw: object) -> None:
        received.append(kw)

    message_added.connect(_receiver)
    try:
        await repo.add_message(
            thread.id,
            role="user",
            parts=[{"kind": "text", "content": "shh"}],
            silent=True,
        )
    finally:
        message_added.disconnect(_receiver)

    assert received == []
    history = await repo.history(thread.id, limit=10)
    assert len(history) == 1

import pytest

from ballast.persistence.thread import (
    InMemoryThreadRepository,
    ThreadClosedError,
    ThreadRepository,
    ThreadStatus,
)


@pytest.fixture
def repo() -> ThreadRepository:
    return InMemoryThreadRepository()


@pytest.mark.asyncio
async def test_create_and_load_thread(repo):
    thread = await repo.create(
        agent="conversation",
        metadata={},
    )
    loaded = await repo.load(thread.id)
    assert loaded.id == thread.id
    assert loaded.agent == "conversation"


@pytest.mark.asyncio
async def test_add_message_and_read_history(repo):
    thread = await repo.create(
        agent="conversation",
        metadata={},
    )
    await repo.add_message(
        thread.id, role="user", parts=[{"kind": "text", "content": "hi"}]
    )
    await repo.add_message(
        thread.id, role="assistant", parts=[{"kind": "text", "content": "hello"}]
    )
    history = await repo.history(thread.id, limit=10)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


@pytest.mark.asyncio
async def test_history_respects_limit_oldest_first(repo):
    thread = await repo.create(
        agent="conversation",
        metadata={},
    )
    for i in range(5):
        await repo.add_message(
            thread.id, role="user", parts=[{"kind": "text", "content": f"m{i}"}]
        )
    history = await repo.history(thread.id, limit=3)
    assert len(history) == 3
    # Oldest first
    assert history[0].parts[0]["content"] == "m0"


@pytest.mark.asyncio
async def test_new_thread_status_is_open(repo):
    thread = await repo.create(
        agent="conversation",
        metadata={},
    )
    assert thread.status == ThreadStatus.OPEN
    assert thread.closed_at is None


@pytest.mark.asyncio
async def test_close_thread_transitions_to_closed(repo):
    thread = await repo.create(
        agent="hitl",
        metadata={"gate_kind": "strategy_review"},
    )
    closed = await repo.close(thread.id)
    assert closed.status == ThreadStatus.CLOSED
    assert closed.closed_at is not None
    # Reload and verify state persisted
    loaded = await repo.load(thread.id)
    assert loaded.status == ThreadStatus.CLOSED


@pytest.mark.asyncio
async def test_add_message_to_closed_thread_raises(repo):
    thread = await repo.create(
        agent="conversation",
        metadata={},
    )
    await repo.close(thread.id)
    with pytest.raises(ThreadClosedError):
        await repo.add_message(
            thread.id, role="user", parts=[{"kind": "text", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_agent_accepts_custom_app_string(repo):
    """``agent`` is a free-form registry key — apps pass custom strings freely."""
    thread = await repo.create(
        agent="hitl:strategy_review",  # custom domain-specific agent
        metadata={"wave_id": "abc"},
    )
    assert thread.agent == "hitl:strategy_review"
    loaded = await repo.load(thread.id)
    assert loaded is not None
    assert loaded.agent == "hitl:strategy_review"


@pytest.mark.asyncio
async def test_history_is_linear_ordered_by_created_at(repo):
    """Messages come back in insertion order — flat list, no tree."""
    thread = await repo.create(agent="conversation", metadata={})
    m1 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "1"}],
    )
    m2 = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "2"}],
    )
    m3 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "3"}],
    )
    history = await repo.history(thread.id)
    assert [m.id for m in history] == [m1.id, m2.id, m3.id]


@pytest.mark.asyncio
async def test_add_message_with_id_is_idempotent(repo):
    """Re-adding the same client-supplied id returns the existing row."""
    thread = await repo.create(agent="conversation", metadata={})
    first = await repo.add_message(
        thread.id, role="user", id="client-123",
        parts=[{"type": "text", "text": "hi"}],
    )
    again = await repo.add_message(
        thread.id, role="user", id="client-123",
        parts=[{"type": "text", "text": "different but ignored"}],
    )
    assert again.id == first.id
    assert again.parts == first.parts  # original parts kept
    assert len(await repo.history(thread.id)) == 1


@pytest.mark.asyncio
async def test_delete_messages_drops_only_listed_ids(repo):
    """``delete_messages`` removes specified ids; unknown ids ignored."""
    thread = await repo.create(agent="conversation", metadata={})
    m1 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "1"}],
    )
    m2 = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "2"}],
    )
    m3 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "3"}],
    )
    await repo.delete_messages(thread.id, ids=[m2.id, "nonexistent"])
    history = await repo.history(thread.id)
    assert [m.id for m in history] == [m1.id, m3.id]

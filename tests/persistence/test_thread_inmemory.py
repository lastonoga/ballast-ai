import pytest

from pydantic_ai_stateflow.persistence.thread import (
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
async def test_history_walks_active_branch_picking_newest_sibling(repo):
    """Regenerated branches are siblings; history returns the newest path.

    Tree shape:

        u1 - a1 - u2 - a2_old
                    \- a2_new   <- newer => active

    Active branch: [u1, a1, u2, a2_new]. ``a2_old`` is preserved in
    storage (accessible via ``siblings``) but skipped on the linear
    history walk.
    """
    import asyncio

    thread = await repo.create(agent="conversation", metadata={})

    u1 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "hi"}],
        parent_id=None,
    )
    a1 = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "hello"}],
        parent_id=u1.id,
    )
    u2 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "again"}],
        parent_id=a1.id,
    )
    a2_old = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "v1"}],
        parent_id=u2.id,
    )
    await asyncio.sleep(0.001)
    a2_new = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "v2"}],
        parent_id=u2.id,
    )

    branch = await repo.history(thread.id)
    assert [m.id for m in branch] == [u1.id, a1.id, u2.id, a2_new.id]
    assert a2_old.id not in {m.id for m in branch}

    sibs = await repo.siblings(a2_new.id)
    assert {s.id for s in sibs} == {a2_old.id, a2_new.id}


@pytest.mark.asyncio
async def test_history_falls_back_to_linear_for_legacy_null_parents(repo):
    """Pre-branching threads (all parent_id=NULL) still render in order."""
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

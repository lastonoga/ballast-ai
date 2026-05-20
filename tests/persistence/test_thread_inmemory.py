from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.thread import (
    InMemoryThreadRepository,
    ThreadClosedError,
    ThreadRepository,
    ThreadStatus,
)


@pytest.fixture
def tenant_id():
    return uuid4()


@pytest.fixture
def other_tenant_id():
    return uuid4()


@pytest.fixture
def repo() -> ThreadRepository:
    return InMemoryThreadRepository()


@pytest.mark.asyncio
async def test_create_and_load_thread(repo, tenant_id):
    thread = await repo.create(
        agent="conversation",
        metadata={},
        actor_id="founder-1",
        tenant_id=tenant_id,
    )
    loaded = await repo.load(thread.id, tenant_id=tenant_id)
    assert loaded.id == thread.id
    assert loaded.actor_id == "founder-1"


@pytest.mark.asyncio
async def test_load_returns_none_for_wrong_tenant(repo, tenant_id, other_tenant_id):
    thread = await repo.create(
        agent="hitl",
        metadata={"gate_kind": "x"},
        actor_id="a",
        tenant_id=tenant_id,
    )
    result = await repo.load(thread.id, tenant_id=other_tenant_id)
    assert result is None


@pytest.mark.asyncio
async def test_add_message_and_read_history(repo, tenant_id):
    thread = await repo.create(
        agent="conversation",
        metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id, role="user", parts=[{"kind": "text", "content": "hi"}], tenant_id=tenant_id
    )
    await repo.add_message(
        thread.id, role="assistant", parts=[{"kind": "text", "content": "hello"}], tenant_id=tenant_id
    )
    history = await repo.history(thread.id, tenant_id=tenant_id, limit=10)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


@pytest.mark.asyncio
async def test_history_respects_limit_oldest_first(repo, tenant_id):
    thread = await repo.create(
        agent="conversation",
        metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    for i in range(5):
        await repo.add_message(
            thread.id, role="user", parts=[{"kind": "text", "content": f"m{i}"}], tenant_id=tenant_id
        )
    history = await repo.history(thread.id, tenant_id=tenant_id, limit=3)
    assert len(history) == 3
    # Oldest first
    assert history[0].parts[0]["content"] == "m0"


@pytest.mark.asyncio
async def test_history_cross_tenant_isolation(repo, tenant_id, other_tenant_id):
    """Adding a message to another tenant's thread must fail safely."""
    thread = await repo.create(
        agent="conversation",
        metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    with pytest.raises(KeyError):
        await repo.add_message(
            thread.id, role="user", parts=[], tenant_id=other_tenant_id
        )


@pytest.mark.asyncio
async def test_new_thread_status_is_open(repo, tenant_id):
    thread = await repo.create(
        agent="conversation",
        metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    assert thread.status == ThreadStatus.OPEN
    assert thread.closed_at is None


@pytest.mark.asyncio
async def test_close_thread_transitions_to_closed(repo, tenant_id):
    thread = await repo.create(
        agent="hitl",
        metadata={"gate_kind": "strategy_review"},
        actor_id="a",
        tenant_id=tenant_id,
    )
    closed = await repo.close(thread.id, tenant_id=tenant_id)
    assert closed.status == ThreadStatus.CLOSED
    assert closed.closed_at is not None
    # Reload and verify state persisted
    loaded = await repo.load(thread.id, tenant_id=tenant_id)
    assert loaded.status == ThreadStatus.CLOSED


@pytest.mark.asyncio
async def test_add_message_to_closed_thread_raises(repo, tenant_id):
    thread = await repo.create(
        agent="conversation",
        metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.close(thread.id, tenant_id=tenant_id)
    with pytest.raises(ThreadClosedError):
        await repo.add_message(
            thread.id, role="user", parts=[{"kind": "text", "content": "hi"}],
            tenant_id=tenant_id,
        )


@pytest.mark.asyncio
async def test_close_cross_tenant_raises(repo, tenant_id, other_tenant_id):
    thread = await repo.create(
        agent="conversation",
        metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    with pytest.raises(KeyError):
        await repo.close(thread.id, tenant_id=other_tenant_id)


@pytest.mark.asyncio
async def test_agent_accepts_custom_app_string(repo, tenant_id):
    """``agent`` is a free-form registry key — apps pass custom strings freely."""
    thread = await repo.create(
        agent="hitl:strategy_review",  # ← custom domain-specific agent
        metadata={"wave_id": "abc"},
        actor_id="founder-1",
        tenant_id=tenant_id,
    )
    assert thread.agent == "hitl:strategy_review"
    loaded = await repo.load(thread.id, tenant_id=tenant_id)
    assert loaded is not None
    assert loaded.agent == "hitl:strategy_review"


@pytest.mark.asyncio
async def test_history_walks_active_branch_picking_newest_sibling(
    repo, tenant_id,
):
    """Regenerated branches are siblings; history returns the newest path.

    Tree shape:

        u1 ─ a1 ─ u2 ─ a2_old
                    └─ a2_new   ← newer ⇒ active

    Active branch: [u1, a1, u2, a2_new]. ``a2_old`` is preserved in
    storage (accessible via ``siblings``) but skipped on the linear
    history walk.
    """
    import asyncio

    thread = await repo.create(
        agent="conversation",
        metadata={}, actor_id="a", tenant_id=tenant_id,
    )

    u1 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "hi"}],
        tenant_id=tenant_id, parent_id=None,
    )
    a1 = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "hello"}],
        tenant_id=tenant_id, parent_id=u1.id,
    )
    u2 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "again"}],
        tenant_id=tenant_id, parent_id=a1.id,
    )
    a2_old = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "v1"}],
        tenant_id=tenant_id, parent_id=u2.id,
    )
    # Ensure created_at ordering is unambiguous (in-memory now() is
    # monotonic but tight loops can collide on coarse clocks).
    await asyncio.sleep(0.001)
    a2_new = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "v2"}],
        tenant_id=tenant_id, parent_id=u2.id,
    )

    branch = await repo.history(thread.id, tenant_id=tenant_id)
    assert [m.id for m in branch] == [u1.id, a1.id, u2.id, a2_new.id]
    assert a2_old.id not in {m.id for m in branch}

    # Siblings query surfaces both children of u2.
    sibs = await repo.siblings(a2_new.id, tenant_id=tenant_id)
    assert {s.id for s in sibs} == {a2_old.id, a2_new.id}


@pytest.mark.asyncio
async def test_history_falls_back_to_linear_for_legacy_null_parents(
    repo, tenant_id,
):
    """Pre-branching threads (all parent_id=NULL) still render in order.

    The 0003 migration leaves legacy rows with NULL parent_id; the walker
    detects the all-NULL case and falls back to created_at ordering so
    old conversations don't collapse to a single message.
    """
    thread = await repo.create(
        agent="conversation",
        metadata={}, actor_id="a", tenant_id=tenant_id,
    )
    m1 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "1"}],
        tenant_id=tenant_id,
    )
    m2 = await repo.add_message(
        thread.id, role="assistant", parts=[{"type": "text", "text": "2"}],
        tenant_id=tenant_id,
    )
    m3 = await repo.add_message(
        thread.id, role="user", parts=[{"type": "text", "text": "3"}],
        tenant_id=tenant_id,
    )
    history = await repo.history(thread.id, tenant_id=tenant_id)
    assert [m.id for m in history] == [m1.id, m2.id, m3.id]

from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.thread import (
    InMemoryThreadRepository,
    ThreadClosedError,
    ThreadPurpose,
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
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
        actor_id="founder-1",
        tenant_id=tenant_id,
    )
    loaded = await repo.load(thread.id, tenant_id=tenant_id)
    assert loaded.id == thread.id
    assert loaded.actor_id == "founder-1"


@pytest.mark.asyncio
async def test_load_returns_none_for_wrong_tenant(repo, tenant_id, other_tenant_id):
    thread = await repo.create(
        purpose=ThreadPurpose.HITL.value,
        purpose_metadata={"gate_kind": "x"},
        actor_id="a",
        tenant_id=tenant_id,
    )
    result = await repo.load(thread.id, tenant_id=other_tenant_id)
    assert result is None


@pytest.mark.asyncio
async def test_add_message_and_read_history(repo, tenant_id):
    thread = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
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
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
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
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
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
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    assert thread.status == ThreadStatus.OPEN
    assert thread.closed_at is None


@pytest.mark.asyncio
async def test_close_thread_transitions_to_closed(repo, tenant_id):
    thread = await repo.create(
        purpose=ThreadPurpose.HITL.value,
        purpose_metadata={"gate_kind": "strategy_review"},
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
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
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
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    with pytest.raises(KeyError):
        await repo.close(thread.id, tenant_id=other_tenant_id)


@pytest.mark.asyncio
async def test_purpose_accepts_custom_app_string(repo, tenant_id):
    """ThreadPurpose enum is suggestive — apps pass custom strings freely."""
    thread = await repo.create(
        purpose="hitl:strategy_review",  # ← custom domain-specific purpose
        purpose_metadata={"wave_id": "abc"},
        actor_id="founder-1",
        tenant_id=tenant_id,
    )
    assert thread.purpose == "hitl:strategy_review"
    loaded = await repo.load(thread.id, tenant_id=tenant_id)
    assert loaded is not None
    assert loaded.purpose == "hitl:strategy_review"

import pytest

from pydantic_ai_stateflow.persistence.outbox import (
    InMemoryOutboxRepository,
    OutboxRepository,
)


@pytest.mark.asyncio
async def test_enqueue_and_list_undelivered():
    repo: OutboxRepository = InMemoryOutboxRepository()
    await repo.enqueue(
        event_type="OrderCreated",
        payload={"order_id": "abc", "amount": 100},
    )
    rows = await repo.list_undelivered(limit=10)
    assert len(rows) == 1
    assert rows[0].event_type == "OrderCreated"
    assert rows[0].payload == {"order_id": "abc", "amount": 100}


@pytest.mark.asyncio
async def test_mark_delivered_removes_from_undelivered_list():
    repo: OutboxRepository = InMemoryOutboxRepository()
    await repo.enqueue(event_type="E", payload={})
    [row] = await repo.list_undelivered(limit=10)
    await repo.mark_delivered(row.id)
    assert await repo.list_undelivered(limit=10) == []

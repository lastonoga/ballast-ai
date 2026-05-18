from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.outbox import (
    InMemoryOutboxRepository,
    OutboxRepository,
)


@pytest.mark.asyncio
async def test_enqueue_and_list_undelivered():
    repo: OutboxRepository = InMemoryOutboxRepository()
    tenant_id = uuid4()
    await repo.enqueue(
        event_type="OrderCreated",
        payload={"order_id": "abc", "amount": 100},
        tenant_id=tenant_id,
    )
    rows = await repo.list_undelivered(tenant_id=tenant_id, limit=10)
    assert len(rows) == 1
    assert rows[0].event_type == "OrderCreated"
    assert rows[0].payload == {"order_id": "abc", "amount": 100}


@pytest.mark.asyncio
async def test_mark_delivered_removes_from_undelivered_list():
    repo: OutboxRepository = InMemoryOutboxRepository()
    tenant_id = uuid4()
    await repo.enqueue(event_type="E", payload={}, tenant_id=tenant_id)
    [row] = await repo.list_undelivered(tenant_id=tenant_id, limit=10)
    await repo.mark_delivered(row.id, tenant_id=tenant_id)
    assert await repo.list_undelivered(tenant_id=tenant_id, limit=10) == []


@pytest.mark.asyncio
async def test_undelivered_is_per_tenant():
    repo: OutboxRepository = InMemoryOutboxRepository()
    t1, t2 = uuid4(), uuid4()
    await repo.enqueue(event_type="E1", payload={}, tenant_id=t1)
    await repo.enqueue(event_type="E2", payload={}, tenant_id=t2)
    rows_t1 = await repo.list_undelivered(tenant_id=t1, limit=10)
    assert len(rows_t1) == 1
    assert rows_t1[0].event_type == "E1"

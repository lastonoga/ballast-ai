import pytest

from ballast.persistence.outbox import SqlOutboxRepository


@pytest.mark.asyncio
async def test_enqueue_and_list_undelivered_postgres(session_factory):
    async with session_factory() as session, session.begin():
        repo = SqlOutboxRepository(session)
        await repo.enqueue(event_type="OrderCreated", payload={"x": 1})

    async with session_factory() as s:
        repo2 = SqlOutboxRepository(s)
        rows = await repo2.list_undelivered(limit=10)
        assert any(r.event_type == "OrderCreated" for r in rows)


@pytest.mark.asyncio
async def test_mark_delivered_postgres(session_factory):
    async with session_factory() as session, session.begin():
        repo = SqlOutboxRepository(session)
        await repo.enqueue(event_type="E", payload={})

    async with session_factory() as s:
        repo2 = SqlOutboxRepository(s)
        [row] = await repo2.list_undelivered(limit=10)
        await repo2.mark_delivered(row.id)
        await s.commit()

    async with session_factory() as s:
        repo3 = SqlOutboxRepository(s)
        assert await repo3.list_undelivered(limit=10) == []

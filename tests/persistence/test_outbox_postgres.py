import pytest

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork
from pydantic_ai_stateflow.persistence.outbox import PostgresOutboxRepository


@pytest.mark.asyncio
async def test_enqueue_and_list_undelivered_postgres(session_factory):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresOutboxRepository(uow.session)
        await repo.enqueue(event_type="OrderCreated", payload={"x": 1})

    async with session_factory() as s:
        repo2 = PostgresOutboxRepository(s)
        rows = await repo2.list_undelivered(limit=10)
        assert any(r.event_type == "OrderCreated" for r in rows)


@pytest.mark.asyncio
async def test_mark_delivered_postgres(session_factory):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresOutboxRepository(uow.session)
        await repo.enqueue(event_type="E", payload={})

    async with session_factory() as s:
        repo2 = PostgresOutboxRepository(s)
        [row] = await repo2.list_undelivered(limit=10)
        await repo2.mark_delivered(row.id)
        await s.commit()

    async with session_factory() as s:
        repo3 = PostgresOutboxRepository(s)
        assert await repo3.list_undelivered(limit=10) == []

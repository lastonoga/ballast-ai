from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork, UnitOfWork


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    # Only create SQLite-compatible tables (JSONB tables require Postgres).
    sqlite_tables = [
        t
        for t in SQLModel.metadata.sorted_tables
        if t.name not in (
            "threads", "messages", "outbox",
            "hitl_blocking_requirements", "hitl_decisions", "hitl_authz_denials",
        )
    ]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: SQLModel.metadata.create_all(c, tables=sqlite_tables))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_uow_commits_on_clean_exit(session_factory):
    uow: UnitOfWork = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        pass


@pytest.mark.asyncio
async def test_uow_rollbacks_on_exception(session_factory):
    uow: UnitOfWork = SqlAlchemyUnitOfWork(session_factory)
    with pytest.raises(RuntimeError, match="boom"):
        async with uow:
            raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_uow_explicit_commit_inside_context(session_factory):
    uow: UnitOfWork = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        await uow.commit()


@pytest.mark.asyncio
async def test_uow_protocol_signature_is_satisfied_by_concrete(session_factory):
    instance = SqlAlchemyUnitOfWork(session_factory)
    assert isinstance(instance, UnitOfWork)

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork
from pydantic_ai_stateflow.providers import PersistenceProvider
from pydantic_ai_stateflow.runtime.container import DefaultContainer


@pytest.mark.asyncio
async def test_persistence_provider_binds_session_factory_and_uow_factory() -> None:
    container = DefaultContainer()
    provider = PersistenceProvider("sqlite+aiosqlite:///:memory:")
    await provider.register(container)

    # sessionmaker binding resolves
    session_factory = container.get(async_sessionmaker)
    assert isinstance(session_factory, async_sessionmaker)

    # UoW binding resolves to the factory callable
    uow_factory = container.get(SqlAlchemyUnitOfWork)
    assert callable(uow_factory)


@pytest.mark.asyncio
async def test_persistence_provider_binds_uow_factory_callable() -> None:
    container = DefaultContainer()
    provider = PersistenceProvider("sqlite+aiosqlite:///:memory:")
    await provider.register(container)

    # container.get(SqlAlchemyUnitOfWork) returns the FACTORY, not a UoW
    uow_factory = container.get(SqlAlchemyUnitOfWork)
    assert callable(uow_factory)

    # Calling the factory returns a fresh SqlAlchemyUnitOfWork each time
    uow1 = uow_factory()
    uow2 = uow_factory()
    assert isinstance(uow1, SqlAlchemyUnitOfWork)
    assert isinstance(uow2, SqlAlchemyUnitOfWork)
    # Each call produces a distinct instance
    assert uow1 is not uow2

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork
from pydantic_ai_stateflow.runtime.container import Container


class PersistenceProvider:
    """Binds sessionmaker and UoW factory."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def register(self, container: Container) -> None:
        engine = create_async_engine(self.dsn)
        sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        container.bind(async_sessionmaker, lambda _: sessionmaker)

        def _uow_factory() -> SqlAlchemyUnitOfWork:
            return SqlAlchemyUnitOfWork(sessionmaker)

        # Bind UoW *factory* — callers call it to get fresh UoW
        container.bind(SqlAlchemyUnitOfWork, lambda _: _uow_factory, singleton=True)  # type: ignore[arg-type, return-value]

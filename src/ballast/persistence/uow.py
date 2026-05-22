from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@runtime_checkable
class UnitOfWork(Protocol):
    """Hides SQLAlchemy AsyncSession from Pattern signatures (per spec 4A.0.5).

    Use as an async context manager. On clean exit: commit. On exception:
    rollback. `commit()` is also exposed for callers needing mid-transaction
    commit (e.g. transactional outbox).
    """

    async def __aenter__(self) -> UnitOfWork: ...
    async def __aexit__(self, *exc_info: Any) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


class SqlAlchemyUnitOfWork:
    """Concrete UoW backed by a SQLAlchemy async sessionmaker.

    The session is exposed as `self.session` only inside `persistence/*`
    modules — Patterns / Capabilities must depend on the UnitOfWork
    Protocol, not this concrete class.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork must be entered before accessing session")
        return self._session

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        self._session = self._session_factory()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None

    async def commit(self) -> None:
        if self._session is not None:
            await self._session.commit()

    async def rollback(self) -> None:
        if self._session is not None:
            await self._session.rollback()

"""Shared fixtures for integration tests.

Provides the same PG container + session_factory fixtures as
tests/persistence/conftest.py so the integration smoke test has access to a
real Postgres session without needing pytest_plugins (which is not supported
in non-root conftest files).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

import pydantic_ai_stateflow.persistence.hitl.domain  # noqa: F401
import pydantic_ai_stateflow.persistence.outbox.domain  # noqa: F401
import pydantic_ai_stateflow.persistence.thread.domain  # noqa: F401


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception:
        return False
    return True


_DOCKER_OK = _docker_available()


@pytest.fixture(scope="session")
def pg_container() -> Iterator[PostgresContainer]:
    if not _DOCKER_OK:
        pytest.skip("Docker daemon not available — skipping testcontainers PG tests")
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def pg_dsn(pg_container: PostgresContainer) -> str:
    raw: str = pg_container.get_connection_url()
    without_psycopg2 = re.sub(r"^postgresql\+psycopg2://", "postgresql+asyncpg://", raw)
    return re.sub(r"^postgresql://", "postgresql+asyncpg://", without_psycopg2)


@pytest.fixture(scope="session")
def pg_dsn_sync(pg_dsn: str) -> str:
    return pg_dsn


def _make_engine(pg_dsn: str) -> AsyncEngine:
    return create_async_engine(
        pg_dsn,
        echo=False,
        connect_args={"server_settings": {"TimeZone": "UTC"}},
    )


@pytest.fixture(scope="session")
def create_all_tables(pg_dsn: str) -> None:
    import asyncio

    async def _create() -> None:
        engine = _make_engine(pg_dsn)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())


@pytest.fixture
async def session_factory(
    pg_dsn: str, create_all_tables: None
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = _make_engine(pg_dsn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()

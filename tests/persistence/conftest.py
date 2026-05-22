"""Session-scoped Postgres fixtures via testcontainers.

Fixtures:
    pg_container   — PostgresContainer (session)
    pg_dsn         — asyncpg-compatible DSN   (session)
    pg_dsn_sync    — psycopg2-compatible DSN  (session, retained for API compat)
    create_all_tables — creates all SQLModel tables once (session, autouse)
    session_factory — async_sessionmaker per test (function)

Note: test_uow.py defines its own ``session_factory`` fixture that uses
SQLite. Because module-level fixtures shadow conftest fixtures, test_uow.py
is unaffected by this conftest.
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

# ── Import all persistence modules so SQLModel.metadata is fully populated ──
import ballast.persistence.hitl.domain  # noqa: F401
import ballast.persistence.outbox.domain  # noqa: F401
import ballast.persistence.thread.domain  # noqa: F401

# ── Session-scoped container & DSN fixtures ──────────────────────────────────


def _docker_available() -> bool:
    """Detect whether a Docker daemon is reachable.

    Returns False if Docker is not running OR not installed — in which case
    tests requiring testcontainers will be skipped, not fail.
    """
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
    """Return an asyncpg-compatible DSN (postgresql+asyncpg://)."""
    raw: str = pg_container.get_connection_url()
    # testcontainers returns psycopg2 or postgresql scheme; normalise to asyncpg.
    without_psycopg2 = re.sub(r"^postgresql\+psycopg2://", "postgresql+asyncpg://", raw)
    return re.sub(r"^postgresql://", "postgresql+asyncpg://", without_psycopg2)


@pytest.fixture(scope="session")
def pg_dsn_sync(pg_dsn: str) -> str:
    """Return a psycopg2-compatible DSN alias (kept for API compat; same as pg_dsn here)."""
    # We only have asyncpg installed, so we just echo pg_dsn.
    return pg_dsn


def _make_engine(pg_dsn: str) -> AsyncEngine:
    """Create an async engine with asyncpg settings for timezone-aware datetimes."""
    return create_async_engine(
        pg_dsn,
        echo=False,
        connect_args={"server_settings": {"TimeZone": "UTC"}},
    )


@pytest.fixture(scope="session")
def create_all_tables(pg_dsn: str) -> None:
    """Create all SQLModel tables once per test session using a sync run via asyncpg.

    We run this synchronously with asyncio.run() because pytest session fixtures
    cannot be async in older pytest-asyncio setups.
    """
    import asyncio

    async def _create() -> None:
        engine = _make_engine(pg_dsn)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())


# ── Function-scoped async sessionmaker ───────────────────────────────────────


@pytest.fixture
async def session_factory(
    pg_dsn: str, create_all_tables: None
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Async sessionmaker for one test; disposes engine on teardown.

    Depends on `create_all_tables` to ensure schema exists. Tests that don't
    use this fixture (e.g. test_uow.py with its own SQLite session_factory,
    or test_thread_models which just inspects metadata) don't pay the
    Docker / testcontainers cost.
    """
    engine = _make_engine(pg_dsn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()

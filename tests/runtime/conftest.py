"""Fixtures for the runtime test package.

Provides the ``pg_dsn`` fixture required by ``test_dbos_workflow_smoke.py``.
These fixtures are intentionally minimal — they only provide Docker/Postgres
access; the heavy SQLModel schema setup from ``tests.persistence.conftest``
is NOT imported here (DBOS manages its own system tables via its own migration).

Intentionally NOT using ``pytest_plugins = ["tests.persistence.conftest"]``
because that triggers a "Plugin already registered" error when the full suite
is collected (pytest auto-discovers persistence/conftest.py as a conftest and
loading it again via pytest_plugins causes a double-registration error).
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio
from dbos import DBOS, DBOSConfig
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]


def _docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception:
        return False
    return True


_DOCKER_OK = _docker_available()


@pytest.fixture(scope="session")
def runtime_pg_container() -> Iterator[PostgresContainer]:
    """Session-scoped Postgres container for runtime tests.

    Named ``runtime_pg_container`` (not ``pg_container``) to avoid fixture
    name collision with the session-scoped container in persistence/conftest.py.
    """
    if not _DOCKER_OK:
        pytest.skip("Docker daemon not available — skipping DBOS smoke tests")
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def pg_dsn(runtime_pg_container: PostgresContainer) -> str:
    """asyncpg-compatible DSN for runtime tests."""
    raw: str = runtime_pg_container.get_connection_url()
    without_psycopg2 = re.sub(r"^postgresql\+psycopg2://", "postgresql+asyncpg://", raw)
    return re.sub(r"^postgresql://", "postgresql+asyncpg://", without_psycopg2)


# ---------------------------------------------------------------------------
# DBOS + SQLite runtime (no Docker) for @DBOS.workflow / step smoke tests.
#
# Mirrors ``tests/patterns/conftest.py`` — kept duplicated rather than
# shared because pytest_plugins-importing the other conftest triggers
# "Plugin already registered" when the full suite runs.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dbos_sqlite_runtime() -> Iterator[type[DBOS]]:
    """Module-scoped DBOS runtime backed by ephemeral SQLite."""
    tmpdir = tempfile.mkdtemp(prefix="dbos-runtime-")
    db_path = Path(tmpdir) / "dbos.sqlite"
    DBOS(
        config=DBOSConfig(
            name="stateflow-runtime-test",
            system_database_url=f"sqlite:///{db_path}",
        ),
    )
    DBOS.launch()
    try:
        yield DBOS
    finally:
        DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(
    dbos_sqlite_runtime: type[DBOS],
) -> AsyncIterator[None]:
    """Per-test fresh ``ThreadPoolExecutor`` for DBOS.

    pytest-asyncio closes the event loop between tests, which shuts
    down DBOS's cached executor. Without this swap the second async
    test that uses DBOS dies with "cannot schedule new futures after
    shutdown."
    """
    from dbos._dbos import _get_dbos_instance

    dbos = _get_dbos_instance()
    fresh = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dbos-test-")
    dbos._executor_field = fresh
    yield

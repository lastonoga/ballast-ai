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
from collections.abc import Iterator

import pytest
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

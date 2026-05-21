"""DBOS bootstrap for api tests that exercise workflow endpoints.

Mirrors ``tests/patterns/conftest.py`` and ``tests/runtime/conftest.py``
— intentionally duplicated rather than imported via ``pytest_plugins``
to avoid the "Plugin already registered" double-registration error when
the full suite is collected.
"""
from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio
from dbos import DBOS, DBOSConfig


@pytest.fixture(scope="module")
def dbos_runtime() -> Iterator[type[DBOS]]:
    """Module-scoped DBOS runtime backed by ephemeral SQLite."""
    tmpdir = tempfile.mkdtemp(prefix="dbos-api-")
    db_path = Path(tmpdir) / "dbos.sqlite"
    DBOS(
        config=DBOSConfig(
            name="stateflow-api-test",
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
    dbos_runtime: type[DBOS],
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

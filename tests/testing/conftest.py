"""Fixtures for ``tests/testing/`` — DBOS + SQLite runtime.

Mirrors ``tests/runtime/conftest.py``: a module-scoped DBOS runtime
backed by ephemeral SQLite, plus a per-test ``fresh_dbos_executor``
that swaps in a fresh ``ThreadPoolExecutor`` so subsequent async
tests don't trip the "cannot schedule new futures after shutdown"
error from pytest-asyncio loop teardown.

Duplicated rather than imported via ``pytest_plugins`` because re-
loading another conftest.py through that mechanism produces a
"Plugin already registered" error during full-suite collection.
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
def dbos_sqlite_runtime() -> Iterator[type[DBOS]]:
    """Module-scoped DBOS runtime backed by ephemeral SQLite."""
    tmpdir = tempfile.mkdtemp(prefix="dbos-testing-")
    db_path = Path(tmpdir) / "dbos.sqlite"
    DBOS(
        config=DBOSConfig(
            name="stateflow-testing-pkg-test",
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
    fresh = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dbos-testing-")
    dbos._executor_field = fresh
    yield

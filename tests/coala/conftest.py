"""DBOS bootstrap for CoALA adapter tests.

Mirrors the pattern in ``tests/patterns/conftest.py``: module-scoped
SQLite system database so the suite runs without Docker; per-test
executor reset to avoid ``cannot schedule new futures after shutdown``
across asyncio loops.
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
    tmp = tempfile.mkdtemp(prefix="dbos-coala-")
    DBOS(
        config=DBOSConfig(
            name="coala-test",
            system_database_url=f"sqlite:///{Path(tmp) / 'dbos.sqlite'}",
        )
    )
    DBOS.launch()
    try:
        yield DBOS
    finally:
        DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(dbos_runtime: type[DBOS]) -> AsyncIterator[None]:
    """Swap DBOS's cached executor with a fresh one per async test."""
    from dbos._dbos import _get_dbos_instance

    dbos = _get_dbos_instance()
    fresh = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dbos-test-")
    dbos._executor_field = fresh
    yield

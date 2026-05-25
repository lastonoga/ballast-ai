"""DBOS bootstrap for map_reduce tests."""
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
    tmp = tempfile.mkdtemp(prefix="dbos-mapreduce-")
    DBOS(config=DBOSConfig(
        name="map-reduce-test",
        system_database_url=f"sqlite:///{Path(tmp)/'dbos.sqlite'}",
    ))
    DBOS.launch()
    try: yield DBOS
    finally: DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(dbos_runtime: type[DBOS]) -> AsyncIterator[None]:
    from dbos._dbos import _get_dbos_instance
    _get_dbos_instance()._executor_field = ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="dbos-test-",
    )
    yield

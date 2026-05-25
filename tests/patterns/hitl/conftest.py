"""DBOS bootstrap for HITL channel tests — mirror of tests/patterns/conftest.py."""
from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio
from dbos import DBOS, DBOSConfig

from ballast.patterns.hitl.channels.ui_card import card_kind_registry


@pytest.fixture(autouse=True)
def _isolate_card_registry() -> Iterator[None]:
    """Snapshot + restore card_kind_registry around each test.

    Prevents cross-module re-registration conflicts when multiple test files
    each define a local _Note class with the same __hitl_kind__.
    """
    snapshot = dict(card_kind_registry)
    try:
        yield
    finally:
        card_kind_registry.clear()
        card_kind_registry.update(snapshot)


@pytest.fixture(scope="module")
def dbos_runtime() -> Iterator[type[DBOS]]:
    tmpdir = tempfile.mkdtemp(prefix="dbos-hitl-")
    db_path = Path(tmpdir) / "dbos.sqlite"
    DBOS(config=DBOSConfig(
        name="stateflow-hitl-test",
        system_database_url=f"sqlite:///{db_path}",
    ))
    DBOS.launch()
    try:
        yield DBOS
    finally:
        DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(dbos_runtime: type[DBOS]) -> AsyncIterator[None]:
    from dbos._dbos import _get_dbos_instance
    dbos = _get_dbos_instance()
    fresh = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dbos-test-")
    dbos._executor_field = fresh
    yield

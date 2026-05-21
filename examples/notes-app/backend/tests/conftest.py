"""Shared pytest fixtures for the notes-app backend tests."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio
from dbos import DBOS, DBOSConfig

from pydantic_ai_stateflow.durable import Durable

from notes_app.notes.repository import InMemoryNoteRepository


@pytest.fixture
def repo() -> InMemoryNoteRepository:
    """Fresh in-memory note repo per test (no cross-test leakage)."""
    return InMemoryNoteRepository()


# ---------------------------------------------------------------------------
# DBOS runtime for tests that exercise @DBOS.workflow code paths
# (the HITLGate.run workflow + the DBOS.send/recv unblock loop).
#
# Mirrors ``tests/patterns/conftest.py`` in the framework: SQLite in a
# tempdir (no Docker), module-scoped runtime, per-test executor swap.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dbos_runtime() -> Iterator[type[DBOS]]:
    """Module-scoped DBOS runtime backed by an ephemeral SQLite file.

    ``destroy_registry=False`` on teardown — the framework's
    ``HITLGate`` is decorated at import time and its registry entries
    must survive across test modules.
    """
    tmpdir = tempfile.mkdtemp(prefix="dbos-notes-app-")
    db_path = Path(tmpdir) / "dbos.sqlite"
    Durable.init(
        DBOSConfig(
            name="notes-app-test",
            system_database_url=f"sqlite:///{db_path}",
        ),
    )
    Durable.launch()
    try:
        yield DBOS
    finally:
        Durable.destroy(destroy_registry=False)


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

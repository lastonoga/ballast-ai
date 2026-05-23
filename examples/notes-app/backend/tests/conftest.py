"""Shared pytest fixtures for the notes-app backend tests."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio
from dbos import DBOS, DBOSConfig

from ballast.durable import Durable
from ballast.events import (
    chat_message_requested,
    helper_thread_created,
    message_added,
)
from ballast.patterns.divergent_convergent.events import (
    divergent_convergent_progress,
)
from ballast.runtime.engine import _reset_ballast_for_tests

from notes_app.repositories.note import InMemoryNoteRepository
from notes_app.workflows.brainstorm import brainstorm_progress


@pytest.fixture(autouse=True)
def _isolate_signals() -> Iterator[None]:
    """Snapshot + restore framework signal receivers around each test.

    Without this, tests that boot the full Ballast app (which connects
    framework defaults via ``EventsProvider``) leak receivers into the
    next test's signal list — leading to duplicate ``message-added`` fires
    or stale closures pointing at torn-down engines.
    """
    snapshots = {
        s: list(s._receivers)
        for s in (
            message_added,
            helper_thread_created,
            chat_message_requested,
            divergent_convergent_progress,
            brainstorm_progress,
        )
    }
    try:
        yield
    finally:
        for s, rec in snapshots.items():
            s._receivers = list(rec)


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> InMemoryNoteRepository:
    """Fresh in-memory note repo per test (no cross-test leakage).

    Also installs the fresh repo as the module-level ``notes_repo``
    singleton so the tools — which now read the singleton directly —
    see this test's repo instead of the import-time global.
    """
    fresh = InMemoryNoteRepository()
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", fresh)
    return fresh


@pytest.fixture(autouse=True)
def _restore_engine_between_tests() -> Iterator[None]:
    """Snapshot the process-wide ``Engine`` and restore it after each test.

    The framework refuses to reassign ``_engine`` to a different
    instance once installed (see ``_set_ballast``). Tests that install a
    fresh Engine (``test_todo_proposal``) and tests that boot the full
    app via ``TestClient(app)`` (``test_smoke``) would otherwise collide
    depending on collection order. Snapshotting on enter + restoring on
    exit keeps the global stable across the suite.
    """
    from ballast.runtime import engine as _engine_mod

    snapshot = _engine_mod._engine
    try:
        yield
    finally:
        _reset_ballast_for_tests()
        if snapshot is not None:
            from ballast.runtime.engine import _set_ballast

            _set_ballast(snapshot)


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

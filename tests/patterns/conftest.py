"""DBOS bootstrap for pattern tests that need a running runtime.

``DivergentConvergent`` declares ``@DBOS.workflow`` / ``@DBOS.step``
inside its run path, so DBOS must be initialised before the wrapped
callables can be invoked. We use a session-scoped temporary SQLite
system database (DBOS supports sqlite natively) so the suite runs
without Docker, and only spin DBOS up for the tests that actually
need it (those that explicitly request ``dbos_runtime``).

Event-loop / executor note: pytest-asyncio (mode=auto) creates a fresh
event loop per async test, and each loop's default executor is shut
down on loop close. DBOS caches its own ``ThreadPoolExecutor`` in
``dbos._executor_field`` and installs it as the running loop's default
via ``_configure_asyncio_thread_pool()``. After the first test's loop
closes, DBOS's cached executor is broken; the next test reinstalls the
broken executor and dies with ``cannot schedule new futures after
shutdown``. Fix: replace ``dbos._executor_field`` with a fresh pool
before every async test that touches DBOS.
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
    tmpdir = tempfile.mkdtemp(prefix="dbos-patterns-")
    db_path = Path(tmpdir) / "dbos.sqlite"
    DBOS(
        config=DBOSConfig(
            name="stateflow-patterns-test",
            system_database_url=f"sqlite:///{db_path}",
        )
    )
    DBOS.launch()
    try:
        yield DBOS
    finally:
        # destroy_registry=False — leave @DBOS.workflow / @DBOS.step
        # registrations intact for subsequent test modules that import
        # patterns (decorators ran at import time and would not be
        # re-applied if their registry entries were wiped).
        DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(
    dbos_runtime: type[DBOS],
) -> AsyncIterator[None]:
    """Swap DBOS's cached executor with a fresh one per async test.

    Tests opt in by depending on this fixture. See module docstring.
    """
    from dbos._dbos import _get_dbos_instance

    dbos = _get_dbos_instance()
    fresh = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dbos-test-")
    dbos._executor_field = fresh
    yield

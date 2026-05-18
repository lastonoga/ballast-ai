"""End-to-end DBOS smoke test: workflow uses Det.uuid_for, replay returns same value.

Requires Docker + Postgres (testcontainers). Skips cleanly otherwise.

API notes for DBOS 2.22.0 (discovered during Task 10):
- ``DBOS(config=<DBOSConfig TypedDict>)`` — constructor takes keyword ``config``.
  Key fields: ``name`` (str), ``database_url`` (deprecated but works for system DB),
  ``notification_listener_polling_interval_sec`` (float, reduces poll latency in tests).
- ``DBOS.launch()`` — sync class method; must be called after constructor.
- ``DBOS.destroy(destroy_registry=True)`` — cleanup; ``destroy_registry=True`` clears
  the global workflow/step decorator registry so other test modules aren't polluted.
- ``DBOS.start_workflow_async(fn, *args)`` — async; returns ``WorkflowHandleAsync``.
  No ``workflow_id`` kwarg — use ``SetWorkflowID(wf_id)`` context manager instead.
- ``SetWorkflowID(wf_id)`` — context manager that sets the id for the next workflow.
- ``WorkflowHandleAsync.get_result()`` — async coroutine (type hint says sync, but
  ``asyncio.iscoroutinefunction`` returns True for the concrete impl).
- ``@DBOS.workflow()`` registration happens at *decoration* time (import time), not
  when ``DBOS(...)`` is called; the function must exist at module level.

Event-loop isolation caveat:
  pytest-asyncio ``asyncio_mode=auto`` creates one event loop *per test function*.
  When a test loop closes, its default thread-pool executor shuts down, which
  kills DBOS's internal ``asyncio.to_thread`` machinery. Running two async tests
  that share a module-scoped DBOS fixture therefore crashes on the second test.

  Work-around: both workflow assertions live inside ONE async test function so
  they run in the same event loop. The fixture initialises DBOS for the whole
  module and destroys it at the end.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from uuid import UUID

import pytest
from dbos import DBOS, DBOSConfig, SetWorkflowID

from pydantic_ai_stateflow.runtime import Det, IdempotencyInput

# ---------------------------------------------------------------------------
# Workflow definition — module-level so @DBOS.workflow() fires at import time.
# ---------------------------------------------------------------------------


@DBOS.workflow()
async def _sample_workflow() -> UUID:
    """Workflow that calls Det.uuid_for — result should be durable."""
    return await Det.uuid_for(IdempotencyInput(namespace="smoke", parts={"x": 1}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync_dsn(pg_dsn: str) -> str:
    """Strip the +asyncpg driver specifier so DBOS can use psycopg."""
    return re.sub(r"^postgresql\+asyncpg://", "postgresql://", pg_dsn)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dbos_runtime(pg_dsn: str) -> Iterator[type[DBOS]]:
    """Initialise + launch DBOS once for the entire module; teardown at end.

    ``destroy_registry=True`` clears the @DBOS.workflow / @DBOS.step global
    registry so it doesn't bleed into other test modules that import DBOS.

    NOTE: All tests in this module that use this fixture MUST run in the
    SAME asyncio event loop — see the module-level docstring for why.
    Concretely: keep all DBOS async calls inside a single async test function.
    """
    sync_url = _sync_dsn(pg_dsn)
    DBOS(
        config=DBOSConfig(
            name="stateflow-smoke-test",
            database_url=sync_url,
            # Reduce poll latency so tests complete quickly:
            notification_listener_polling_interval_sec=0.05,
        )
    )
    DBOS.launch()
    try:
        yield DBOS
    finally:
        DBOS.destroy(destroy_registry=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dbos_workflow_det_uuid_for_and_replay(dbos_runtime: type[DBOS]) -> None:
    """Two assertions in one test to share a single asyncio event loop.

    1. UUID result: workflow returns a valid UUID v5.
    2. Replay determinism: resubmitting the same workflow_id returns the same UUID
       (Critical Fix #1 proof-of-life — Det.uuid_for is @DBOS.step, so its result
       is durably recorded and replayed verbatim, not recomputed).

    Why one function instead of two:
        pytest-asyncio creates a fresh event loop per test function. Closing the
        first loop shuts down its thread-pool executor, which kills DBOS's
        ``asyncio.to_thread`` machinery used by ``start_workflow_async``. Keeping
        both assertions in one function ensures a single continuous event loop.
    """
    # --- Part 1: result is a stable UUID v5 ---------------------------------
    handle_first = await DBOS.start_workflow_async(_sample_workflow)
    result_first = await handle_first.get_result()

    assert isinstance(result_first, UUID), f"Expected UUID, got {type(result_first)}"
    assert result_first.version == 5, f"Expected UUID v5, got v{result_first.version}"

    # --- Part 2: replay returns the same UUID (durability guarantee) --------
    workflow_id = "smoke-test-replay-determinism"

    with SetWorkflowID(workflow_id):
        handle_a = await DBOS.start_workflow_async(_sample_workflow)
    result_a = await handle_a.get_result()

    # Resubmit the same workflow_id — DBOS should return the recorded result.
    with SetWorkflowID(workflow_id):
        handle_b = await DBOS.start_workflow_async(_sample_workflow)
    result_b = await handle_b.get_result()

    assert result_a == result_b, (
        f"Replay returned a different UUID: first={result_a!r}, replay={result_b!r}. "
        "Det.uuid_for must be a @DBOS.step so its result is durably recorded."
    )

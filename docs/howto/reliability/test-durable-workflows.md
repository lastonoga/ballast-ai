# How to test DBOS workflows

**Problem:** Your code uses `@Durable.workflow` / `@Durable.step` / `Durable.recv_async`. Tests need a real DBOS instance — but a full Postgres setup per test is slow. You want a fast, isolated, in-process DBOS that just works in pytest.

**Solution:** Module-scoped DBOS fixture backed by SQLite + per-test fresh executor. Established pattern used by every Ballast pattern test suite.

## Copy this fixture

`tests/your_pattern/conftest.py`:
```python
"""DBOS bootstrap for workflow-bound tests."""
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
    tmp = tempfile.mkdtemp(prefix="dbos-test-")
    DBOS(config=DBOSConfig(
        name="test",
        system_database_url=f"sqlite:///{Path(tmp)/'dbos.sqlite'}",
    ))
    DBOS.launch()
    try:
        yield DBOS
    finally:
        DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(dbos_runtime: type[DBOS]) -> AsyncIterator[None]:
    """Per-test fresh ThreadPoolExecutor so async tasks don't leak across tests."""
    from dbos._dbos import _get_dbos_instance
    _get_dbos_instance()._executor_field = ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="dbos-test-",
    )
    yield
```

That's it. Drop into any test directory; reuse the same pattern.

## Use the fixture in a workflow test

```python
import pytest
from ballast import Durable


@Durable.workflow()
async def my_workflow(x: int) -> int:
    return x * 2


@pytest.mark.asyncio
async def test_workflow_runs(fresh_dbos_executor) -> None:
    result = await my_workflow(5)
    assert result == 10
```

The `fresh_dbos_executor` fixture is a no-arg dependency — just include it in your test signature. DBOS workflow + step decorators now work normally.

## Test step memoisation

```python
counter = {"n": 0}


@Durable.step()
async def expensive_step() -> int:
    counter["n"] += 1
    return counter["n"]


@Durable.workflow()
async def memo_workflow() -> tuple[int, int]:
    a = await expensive_step()
    b = await expensive_step()    # second call within same workflow
    return a, b


@pytest.mark.asyncio
async def test_step_memoised_within_workflow(fresh_dbos_executor) -> None:
    a, b = await memo_workflow()
    # Both calls are different — memoisation is per-call-site, not per-function
    assert (a, b) == (1, 2)
```

For replay tests (crash + restart), you'd inject a controlled exception mid-workflow and re-trigger — more advanced; usually framework-internal.

## Test HITL waiting via Durable.recv_async

```python
import asyncio


@Durable.workflow()
async def waiting_workflow() -> str:
    return await Durable.recv_async(topic="approval:test", timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_hitl_resume(fresh_dbos_executor) -> None:
    # Start workflow in background
    task = asyncio.create_task(waiting_workflow())
    await asyncio.sleep(0.05)    # let workflow reach Durable.recv_async

    # Send the verdict
    await Durable.send_async(topic="approval:test", message="approved")

    result = await task
    assert result == "approved"
```

This pattern tests the full HITL flow end-to-end without real channels or APIs.

## Module-scoped DBOS, function-scoped state

The DBOS instance is `scope="module"` (one per test file). The executor is per-test (function-scoped). This balances setup cost (~50ms for DBOS launch) against isolation (no thread leaks across tests).

If you have tests that genuinely need a fresh DBOS (e.g. testing registration behavior), you can mark them with `scope="function"` on a custom fixture — but the standard pattern works for 99% of cases.

## Run a subset of workflow tests

```
uv run pytest tests/patterns/plan_execute/ -v
```

The module-scoped fixture means all tests in that file share one DBOS — fast.

## Composing with other patterns

```python
from ballast.patterns.plan_execute import PlanAndExecute


@pytest.mark.asyncio
async def test_plan_execute_with_dbos(fresh_dbos_executor) -> None:
    plan = Plan(steps=[
        PlannedStep(id="a", kind="callable", params={"fn_name": "fn_a"}),
    ])
    registry = StepRegistry.with_defaults()
    async def fn_a(*, plan_input, dep_outputs): return "OK"
    registry.register_callable("fn_a", fn_a)

    pattern = PlanAndExecute(planner=_FakePlanner(plan), registry=registry)
    outputs = await pattern.run("input")
    assert outputs == {"a": "OK"}
```

The pattern uses `@Durable.workflow` internally; the fixture provides the runtime.

## When NOT to use this

- **Pure pattern tests with no Durable code** — skip the fixture, just call your async function. e.g. `MapReduce` with callable steps + no agents — no DBOS needed.
- **Capability tests** — capabilities don't need DBOS. Use `TestModel` + `AsyncMock` for `RunContext`.
- **CoALA Unit tests** — direct phase calls don't need DBOS. Only `as_workflow(unit)` does.

## Caveats

- **Don't share SQLite path across modules.** Each `dbos_runtime` fixture creates a fresh `tempfile.mkdtemp()` — no collision.
- **`destroy(destroy_registry=False)`** — keeps the workflow registry across runs (needed for module-scoped fixture). Don't set `destroy_registry=True` unless you mean it.
- **Thread pool leak warning.** If a test starts background DBOS work and doesn't await it, the per-test fixture cleans up. But if your code holds references to the old executor, you get a `RuntimeError` on subsequent fixture invocations. Always `await` your async tasks.
- **No Postgres in tests.** All tests use SQLite. Production behavior differences (e.g. concurrent locking) need separate integration tests against a real Postgres — usually a small `tests/integration_postgres/` directory marked `@pytest.mark.integration`.

## Related

- [test-without-real-llm.md](test-without-real-llm.md) — TestModel patterns
- [test-coala-units.md](test-coala-units.md) — when you DON'T need DBOS
- Reference: `reference/core/durable.md`

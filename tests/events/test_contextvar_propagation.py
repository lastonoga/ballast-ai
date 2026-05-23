"""Does a ContextVar set in a parent workflow body survive into a child
workflow body? Used to decide whether contextvar-based progress
routing is viable across DBOS boundaries."""
from __future__ import annotations

from contextvars import ContextVar

import pytest

from ballast.durable import Durable


_cv: ContextVar[str | None] = ContextVar("test_cv_propagation", default=None)


@pytest.mark.asyncio
async def test_contextvar_visible_in_child_workflow(
    fresh_dbos_executor: None,
) -> None:
    del fresh_dbos_executor
    captured: list[str | None] = []

    @Durable.workflow()
    async def child() -> None:
        captured.append(_cv.get())

    @Durable.workflow()
    async def parent() -> None:
        _cv.set("parent-value")
        await child()

    await parent()
    assert captured == ["parent-value"], (
        f"ContextVar did not propagate to child workflow body. "
        f"Got: {captured!r}"
    )


@pytest.mark.asyncio
async def test_contextvar_visible_in_durable_step(
    fresh_dbos_executor: None,
) -> None:
    del fresh_dbos_executor
    captured: list[str | None] = []

    @Durable.step()
    async def step_fn() -> None:
        captured.append(_cv.get())

    @Durable.workflow()
    async def parent() -> None:
        _cv.set("parent-value")
        await step_fn()

    await parent()
    assert captured == ["parent-value"]


@pytest.mark.xfail(
    reason=(
        "Known limitation: ``Durable.enqueue`` dispatches workers via the "
        "DBOS queue manager (thread-pool worker that imports the function "
        "fresh, no parent-context inheritance). ContextVars do NOT cross "
        "this boundary. Patterns that rely on ``progress_thread_var`` must "
        "emit signals from the WORKFLOW BODY (parent fiber), not from "
        "queue-worker step bodies — which is exactly how "
        "``DivergentConvergent`` is structured today."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_contextvar_visible_in_queue_worker(
    fresh_dbos_executor: None,
) -> None:
    """Critical case for DivergentConvergent — branches run via Durable.enqueue."""
    del fresh_dbos_executor
    from dbos import Queue

    captured: list[str | None] = []
    queue = Queue("test_cv_queue", concurrency=2)

    @Durable.step()
    async def worker(idx: int) -> int:
        captured.append(_cv.get())
        return idx

    @Durable.workflow()
    async def parent() -> None:
        _cv.set("parent-value")
        handle = await Durable.enqueue(queue, worker, 1)
        await handle.get_result()

    await parent()
    assert captured == ["parent-value"], (
        f"ContextVar did not propagate to queue worker. Got: {captured!r}"
    )

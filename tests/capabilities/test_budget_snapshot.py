"""BudgetGuard.snapshot() — bridge for OnBudgetThreshold drift strategy."""
from __future__ import annotations

import pytest

from ballast.capabilities.budget import BudgetGuard


@pytest.mark.asyncio
async def test_snapshot_returns_counters_and_limits() -> None:
    bg = BudgetGuard(max_iterations=10, max_input_tokens=1000, max_output_tokens=500)
    per_run = await bg.for_run(ctx=None)  # type: ignore[arg-type]
    snap = per_run.snapshot()
    assert snap == {
        "iterations": 0, "max_iterations": 10,
        "input_tokens": 0, "max_input_tokens": 1000,
        "output_tokens": 0, "max_output_tokens": 500,
    }


@pytest.mark.asyncio
async def test_snapshot_updates_after_request() -> None:
    bg = BudgetGuard(max_iterations=10, max_input_tokens=1000)
    per_run = await bg.for_run(ctx=None)  # type: ignore[arg-type]

    # Simulate after_model_request bookkeeping
    per_run._iterations = 3
    per_run._input_tokens = 250
    per_run._output_tokens = 100

    snap = per_run.snapshot()
    assert snap["iterations"] == 3
    assert snap["input_tokens"] == 250
    assert snap["output_tokens"] == 100
    assert snap["max_output_tokens"] is None  # unset

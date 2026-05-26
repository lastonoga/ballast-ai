"""PlanExecutionError — raised under FailLoud."""
from __future__ import annotations

import pytest

from ballast.errors import BallastError
from ballast.patterns.plan_execute._errors import PlanExecutionError
from ballast.patterns.plan_execute._plan import PlannedStep


def test_plan_execution_error_subclass_of_ballast_error() -> None:
    assert issubclass(PlanExecutionError, BallastError)


def test_plan_execution_error_has_code() -> None:
    assert PlanExecutionError.code == "BALLAST_PLAN_EXECUTION"


def test_plan_execution_error_carries_failed_step_and_partial_outputs() -> None:
    step = PlannedStep(id="x", kind="llm")
    exc = PlanExecutionError(
        "step failed",
        failed_step=step,
        partial_outputs={"a": "done", "b": 42},
    )
    assert exc.failed_step is step
    assert exc.partial_outputs == {"a": "done", "b": 42}
    assert "step failed" in str(exc)


def test_plan_execution_error_chain_via_from() -> None:
    step = PlannedStep(id="x", kind="llm")
    original = RuntimeError("network down")
    try:
        try:
            raise original
        except RuntimeError as cause:
            raise PlanExecutionError(
                "step x failed",
                failed_step=step,
                partial_outputs={},
            ) from cause
    except PlanExecutionError as exc:
        assert exc.__cause__ is original

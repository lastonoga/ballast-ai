"""FailLoud RePlanPolicy — only built-in in first cut."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._policies import FailLoud
from ballast.patterns.plan_execute._protocols import RePlanPolicy


def test_fail_loud_satisfies_replan_policy_protocol() -> None:
    assert isinstance(FailLoud(), RePlanPolicy)


@pytest.mark.asyncio
async def test_fail_loud_returns_none() -> None:
    plan = Plan(steps=[PlannedStep(id="a", kind="llm")])
    out = await FailLoud().on_step_failure(
        plan=plan, failed_step=plan.steps[0],
        error=RuntimeError("oops"), partial_outputs={},
    )
    assert out is None

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.evals import (
    Dataset,
    EvalCase,
    EvalReport,
    SchemaAdherenceScorer,
)


class _Out(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_dataset_evaluate_returns_report_with_per_case_scores():
    cases = [
        EvalCase(name="c1", inputs={"x": 1}, expected={}, metadata={}),
        EvalCase(name="c2", inputs={"x": 2}, expected={}, metadata={}),
    ]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(inputs: dict[str, int]) -> _Out:
        return _Out(text=f"hi-{inputs['x']}")

    report = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer()])
    assert isinstance(report, EvalReport)
    assert len(report.case_scores) == 2
    assert all(0.0 <= cs.score <= 1.0 for cs in report.case_scores)


@pytest.mark.asyncio
async def test_dataset_report_aggregates_mean_per_scorer():
    cases = [EvalCase(name=f"c{i}", inputs={}, expected={}, metadata={}) for i in range(3)]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(_inputs):
        return _Out(text="x")

    rep = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer()])
    assert "SchemaAdherenceScorer" in rep.scorer_means
    assert rep.scorer_means["SchemaAdherenceScorer"] == 1.0


@pytest.mark.asyncio
async def test_dataset_passed_respects_thresholds():
    cases = [EvalCase(name="c1", inputs={}, expected={}, metadata={})]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(_inputs):
        return _Out(text="x")

    rep_ok = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer(threshold=0.5)])
    assert rep_ok.passed is True

    rep_bad = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer(threshold=2.0)])
    assert rep_bad.passed is False


@pytest.mark.asyncio
async def test_dataset_rejects_cross_tenant_metadata():
    """Spec 1.12: Dataset filtered by tenant_id; cross-tenant cases dropped."""
    tid = uuid4()
    other = uuid4()
    cases = [
        EvalCase(name="c1", inputs={}, expected={}, metadata={"tenant_id": str(tid)}),
        EvalCase(name="c2", inputs={}, expected={}, metadata={"tenant_id": str(other)}),
    ]
    ds = Dataset(name="t", tenant_id=tid, cases=cases)
    assert len(ds.cases) == 1
    assert ds.cases[0].name == "c1"


@pytest.mark.asyncio
async def test_dataset_evaluate_captures_runner_exception_as_score_zero():
    cases = [EvalCase(name="c1", inputs={}, expected={}, metadata={})]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(_inputs):
        raise RuntimeError("boom")

    rep = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer()])
    assert rep.case_scores[0].score == 0.0
    assert "boom" in (rep.case_scores[0].error or "")

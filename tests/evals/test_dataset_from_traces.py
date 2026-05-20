from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.evals.traces import (
    InMemoryTraceSource,
    TraceRecord,
    dataset_from_traces,
)


@pytest.mark.asyncio
async def test_dataset_from_traces_filters_by_pattern():
    now = datetime.now(tz=UTC)
    src = InMemoryTraceSource(records=[
        TraceRecord(
            run_id=uuid4(), pattern="reflection",
            inputs={"x": 1}, output={"text": "y"},
            created_at=now, outcome="success",
        ),
        TraceRecord(
            run_id=uuid4(), pattern="mapreduce",
            inputs={"x": 3}, output={"text": "a"},
            created_at=now, outcome="success",
        ),
    ])
    ds = await dataset_from_traces(
        src, pattern="reflection",
        since=now - timedelta(days=1),
    )
    assert ds.name == "reflection-traces"
    assert len(ds.cases) == 1
    assert ds.cases[0].inputs == {"x": 1}


@pytest.mark.asyncio
async def test_dataset_from_traces_excludes_pre_since_records():
    old = datetime.now(tz=UTC) - timedelta(days=10)
    new = datetime.now(tz=UTC)
    src = InMemoryTraceSource(records=[
        TraceRecord(
            run_id=uuid4(), pattern="p",
            inputs={}, output={}, created_at=old, outcome="success",
        ),
        TraceRecord(
            run_id=uuid4(), pattern="p",
            inputs={}, output={}, created_at=new, outcome="success",
        ),
    ])
    ds = await dataset_from_traces(
        src, pattern="p",
        since=datetime.now(tz=UTC) - timedelta(days=1),
    )
    assert len(ds.cases) == 1


@pytest.mark.asyncio
async def test_dataset_from_traces_attaches_run_id_metadata():
    """Spec 1.14 — run_id traceability back to production incident."""
    rid = uuid4()
    src = InMemoryTraceSource(records=[
        TraceRecord(
            run_id=rid, pattern="p",
            inputs={}, output={}, created_at=datetime.now(tz=UTC),
            outcome="success",
        ),
    ])
    ds = await dataset_from_traces(
        src, pattern="p",
        since=datetime.now(tz=UTC) - timedelta(days=1),
    )
    assert ds.cases[0].metadata["run_id"] == str(rid)

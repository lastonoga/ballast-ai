"""Trace ingestion for building eval datasets from production runs (spec 1.14)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.evals.case import EvalCase
from pydantic_ai_stateflow.evals.dataset import Dataset


class TraceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: UUID
    tenant_id: UUID
    pattern: str
    inputs: Any
    output: Any
    created_at: datetime
    outcome: str


@runtime_checkable
class TraceSource(Protocol):
    async def query(
        self,
        *,
        tenant_id: UUID,
        pattern: str | None,
        since: datetime,
    ) -> list[TraceRecord]: ...


class InMemoryTraceSource:
    def __init__(self, records: list[TraceRecord]) -> None:
        self._records = list(records)

    async def query(
        self,
        *,
        tenant_id: UUID,
        pattern: str | None,
        since: datetime,
    ) -> list[TraceRecord]:
        return [
            r for r in self._records
            if r.tenant_id == tenant_id
            and (pattern is None or r.pattern == pattern)
            and r.created_at >= since
        ]


async def dataset_from_traces(
    source: TraceSource,
    *,
    tenant_id: UUID,
    pattern: str | None,
    since: datetime,
    name: str | None = None,
) -> Dataset:
    """Build a Dataset by joining production trace records.

    Spec 1.14 — each production run becomes a reusable eval case;
    `run_id` is preserved in metadata so an eval failure traces back to
    the originating incident.
    """
    records = await source.query(
        tenant_id=tenant_id, pattern=pattern, since=since,
    )
    cases = [
        EvalCase(
            name=f"run-{r.run_id}",
            inputs=r.inputs,
            expected=r.output,
            metadata={
                "run_id": str(r.run_id),
                "tenant_id": str(r.tenant_id),
                "outcome": r.outcome,
                "pattern": r.pattern,
            },
        )
        for r in records
    ]
    return Dataset(
        name=name or f"{pattern or 'all'}-traces",
        tenant_id=tenant_id,
        cases=cases,
    )

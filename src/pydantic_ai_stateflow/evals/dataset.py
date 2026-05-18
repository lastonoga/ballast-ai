from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai_stateflow.evals.case import EvalCase, EvalRunOutput
from pydantic_ai_stateflow.evals.scorer import Scorer

Runner = Callable[[Any], Awaitable[Any]] | Callable[[Any], Any]


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_name: str
    scorer_name: str
    score: float
    error: str | None = None


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_name: str
    case_scores: list[ScoreResult]
    scorer_means: dict[str, float] = Field(default_factory=dict)
    passed: bool = True


class Dataset:
    """A collection of `EvalCase`s scoped to a single tenant.

    Cross-tenant cases (metadata `tenant_id` mismatch) are dropped at
    construction — spec 1.12 forbids cross-tenant eval mixing.
    """

    def __init__(
        self,
        *,
        name: str,
        tenant_id: UUID,
        cases: list[EvalCase],
    ) -> None:
        self.name = name
        self.tenant_id = tenant_id
        self.cases: list[EvalCase] = [
            c for c in cases
            if c.metadata.get("tenant_id") in (None, str(tenant_id))
        ]

    async def evaluate(
        self, runner: Runner, *, evaluators: list[Scorer],
    ) -> EvalReport:
        rows: list[ScoreResult] = []
        for case in self.cases:
            try:
                result = runner(case.inputs)
                if inspect.isawaitable(result):
                    output = await result
                else:
                    output = result
                run_out = EvalRunOutput(output=output, retries=0)
            except Exception as exc:
                run_out = EvalRunOutput(output=None, retries=0, error=str(exc))
            for scorer in evaluators:
                score = await scorer.score(run_out)
                rows.append(ScoreResult(
                    case_name=case.name, scorer_name=scorer.name,
                    score=score, error=run_out.error,
                ))
        means = self._aggregate(rows)
        passed = all(
            means.get(s.name, 0.0) >= s.threshold for s in evaluators
        )
        return EvalReport(
            dataset_name=self.name, case_scores=rows,
            scorer_means=means, passed=passed,
        )

    @staticmethod
    def _aggregate(rows: list[ScoreResult]) -> dict[str, float]:
        by_scorer: dict[str, list[float]] = {}
        for r in rows:
            by_scorer.setdefault(r.scorer_name, []).append(r.score)
        return {k: sum(v) / len(v) for k, v in by_scorer.items() if v}

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ballast.evals.case import EvalRunOutput


@runtime_checkable
class Scorer(Protocol):
    """Pluggable scorer — returns a float in [0.0, 1.0]."""

    threshold: float
    name: str

    async def score(self, run: EvalRunOutput) -> float: ...


class SchemaAdherenceScorer:
    """1.0 if the runner produced a valid BaseModel without retries."""

    name = "SchemaAdherenceScorer"

    def __init__(self, *, threshold: float = 0.95) -> None:
        self.threshold = threshold

    async def score(self, run: EvalRunOutput) -> float:
        if run.error is not None or run.output is None:
            return 0.0
        if run.retries > 0:
            return 0.0
        if not isinstance(run.output, BaseModel):
            return 0.0
        return 1.0

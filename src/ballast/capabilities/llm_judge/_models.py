"""Verdict data shapes — what an :class:`LLMJudge` returns."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel, populate_by_name=True):
    """Result of one direct grading call.

    ``pass_`` / ``score`` mirror pydantic-evals' ``GradingOutput``;
    ``model_used`` + ``latency_ms`` are framework additions so
    production monitoring can attribute cost / latency to the judge.

    The wire-serialised JSON uses ``pass`` (alias) so it round-trips
    cleanly with ``GradingOutput``.
    """

    reason: str
    pass_: bool = Field(validation_alias="pass", serialization_alias="pass")
    score: float
    model_used: str
    latency_ms: int


class PairwiseVerdict(BaseModel):
    """Result of one pairwise comparison.

    ``winner`` ∈ {``"a"``, ``"b"``, ``"tie"``}. ``reason`` is the
    judge's CoT-style justification — same role as
    :attr:`JudgeVerdict.reason`.
    """

    winner: Literal["a", "b", "tie"]
    reason: str
    model_used: str
    latency_ms: int


__all__ = ["JudgeVerdict", "PairwiseVerdict"]

"""``Scored[T, ConfidenceT]`` — generic wrapper carrying value + rationale + confidence.

Use as a tool / agent / pattern output type:

    async def search() -> Scored[list[Note]]: ...
    agent = Agent(output_type=Scored[Summary])

    async def map_fact(item) -> Scored[Fact]: ...

Default ``ConfidenceT`` is ``Literal["low", "medium", "high"]`` — named
labels recommended by the article's stronger principle to avoid the
mean-reversion that affects numeric scales (1-10).

Apps may override:
    Scored[Fact, int]                              # 1-5 numeric scale
    Scored[Fact, Literal["safe", "uncertain"]]     # binary

Built-in helpers in ``_confidence.py`` (``filter_by_min_confidence`` /
``rank_by_confidence`` / ``aggregate_by_confidence`` / ``label_to_float``)
work only with the default ``Confidence`` Literal. Apps with custom
``ConfidenceT`` write their own helpers.

Composition: ``scan_output`` (from ``ballast.grounded``) recurses into
``Scored.value`` automatically — ``Ref[T]`` fields buried inside the
wrapped value are discovered without any special-case wiring.
"""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

try:
    # PEP 696 — `TypeVar(..., default=...)` lands in CPython 3.13.
    # For 3.11/3.12 we use typing_extensions's backport.
    from typing_extensions import TypeVar as _TypeVar
except ImportError:  # pragma: no cover
    from typing import TypeVar as _TypeVar  # type: ignore[assignment]

from ballast.quality.scored._confidence import Confidence


T = TypeVar("T")
ConfidenceT = _TypeVar("ConfidenceT", default=Confidence)


class Scored(BaseModel, Generic[T, ConfidenceT]):
    """Wraps any value with rationale + confidence label.

    See module docstring for usage examples and composition notes.
    """

    model_config = ConfigDict(frozen=True)

    value: T
    rationale: str
    """One-sentence justification — REQUIRED. Forces CoT-style reasoning
    before the LLM commits to a confidence label."""

    confidence: ConfidenceT


__all__ = ["Scored"]

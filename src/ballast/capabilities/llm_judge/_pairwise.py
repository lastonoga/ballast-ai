"""Pairwise comparison — pydantic-evals does not ship it OOB.

Pairwise judging is empirically the most robust mode for subjective
rubrics (tone, helpfulness, style) — single-shot scoring is fragile
because the model has no anchor for what "good" looks like.

This module owns a tiny pydantic-ai Agent + prompt template. Kept
separate from :mod:`ballast.capabilities.llm_judge.judge` because the
direct-mode path delegates entirely to pydantic-evals, while pairwise
is framework-owned — they share nothing operationally.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent

if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings


class _PairwiseGrading(BaseModel):
    """Internal output schema for the pairwise judge agent."""

    reason: str
    winner: Literal["a", "b", "tie"]


_SYSTEM_PROMPT = (
    "You are a strict but fair judge comparing two outputs against a "
    "user-supplied rubric. Always reason step by step before "
    "returning a verdict."
)


def _stringify(value: Any) -> str:
    """Best-effort JSON-friendly stringification for the prompt body."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, default=str)
    except Exception:
        return repr(value)


def build_pairwise_prompt(rubric: str, a: Any, b: Any) -> str:
    """Render the user-facing prompt for one pairwise comparison."""
    return (
        "Compare two outputs against the rubric and choose the "
        "stronger one. If they are equally good or equally bad, "
        "return ``tie``. Always explain your reasoning before the "
        "final verdict.\n\n"
        f"<Rubric>{rubric}</Rubric>\n"
        f"<OutputA>{_stringify(a)}</OutputA>\n"
        f"<OutputB>{_stringify(b)}</OutputB>"
    )


def make_pairwise_agent(
    model_id: str,
    *,
    model_settings: "ModelSettings | None" = None,
) -> Agent[None, _PairwiseGrading]:
    """Construct a fresh pydantic-ai Agent for one pairwise call.

    A fresh agent per call avoids any per-instance state leaking
    across grading invocations.
    """
    return Agent(
        model=model_id,
        output_type=_PairwiseGrading,
        system_prompt=_SYSTEM_PROMPT,
        model_settings=model_settings,
    )


__all__ = [
    "_PairwiseGrading",
    "build_pairwise_prompt",
    "make_pairwise_agent",
]

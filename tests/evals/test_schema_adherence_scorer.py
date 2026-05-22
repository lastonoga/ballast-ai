from __future__ import annotations

import pytest
from pydantic import BaseModel

from ballast.evals import EvalRunOutput, SchemaAdherenceScorer


class _Out(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_scorer_1_for_valid_basemodel_output():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output=_Out(text="x"), retries=0))
    assert score == 1.0


@pytest.mark.asyncio
async def test_scorer_0_when_retries_gt_zero():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output=_Out(text="x"), retries=2))
    assert score == 0.0


@pytest.mark.asyncio
async def test_scorer_0_when_output_is_none():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output=None, retries=0, error="bad"))
    assert score == 0.0


@pytest.mark.asyncio
async def test_scorer_0_for_non_basemodel_output():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output={"text": "x"}, retries=0))
    assert score == 0.0

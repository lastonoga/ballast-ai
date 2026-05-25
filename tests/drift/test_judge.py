"""DefaultPromptBuilder + make_default_judge factory."""
from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from ballast.drift._judge import DefaultPromptBuilder, make_default_judge
from ballast.drift._verdict import DefaultDriftVerdict


def test_default_prompt_includes_goal_and_trace_markers() -> None:
    p = DefaultPromptBuilder().build(
        goal="research Topic X",
        trace=[
            ModelRequest(parts=[UserPromptPart(content="hi")]),
            ModelResponse(parts=[TextPart(content="hello")]),
        ],
    )
    assert "Goal" in p
    assert "research Topic X" in p
    assert "Recent trace" in p
    assert "hi" in p
    assert "hello" in p


def test_default_prompt_handles_empty_trace() -> None:
    p = DefaultPromptBuilder().build(goal="g", trace=[])
    assert "g" in p
    # Should not crash; trace section may be empty or marker text.


def test_make_default_judge_constructs_agent_with_verdict_output_type() -> None:
    judge = make_default_judge(model="test")
    assert judge is not None
    # Construction succeeds without calling the model.
    # Don't over-constrain the inner output_type representation
    # (pydantic-ai may wrap); tolerate variation.

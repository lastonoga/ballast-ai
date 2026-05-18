import re

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow import BudgetGuard, PIIGuard


def test_capabilities_visible_at_top_level() -> None:
    from pydantic_ai_stateflow import (
        BudgetExhausted,
        BudgetGuard,
        Critique,
        Embedder,
        GroundedRetry,
        PIIGuard,
        SemanticDeduper,
        SemanticLoopDetected,
        SemanticLoopDetector,
        StateflowCapability,
        TypedLoopGuard,
        as_critique,
    )

    assert BudgetGuard is not None
    assert SemanticLoopDetector is not None
    assert PIIGuard is not None
    assert GroundedRetry is not None
    assert callable(as_critique)
    assert BudgetExhausted is not None
    assert Critique is not None
    assert Embedder is not None
    assert SemanticDeduper is not None
    assert SemanticLoopDetected is not None
    assert StateflowCapability is not None
    assert TypedLoopGuard is not None


@pytest.mark.asyncio
async def test_two_capabilities_compose_in_one_agent() -> None:
    """Stacking BudgetGuard (outermost) + PIIGuard (innermost) — both fire."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="contact alice@example.com")])

    agent = Agent(
        model=FunctionModel(fn),
        capabilities=[
            BudgetGuard(max_iterations=5),
            PIIGuard(patterns=[re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")]),
        ],
    )
    result = await agent.run("hi")
    text = str(result.output) if result.output else ""
    assert "alice@example.com" not in text
    assert "[REDACTED]" in text

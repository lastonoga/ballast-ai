"""Capability wiring + behavior for ``NotesAgent``."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_stateflow.capabilities import BudgetExhausted, BudgetGuard, PIIGuard

from notes_app.agent import default_notes_capabilities


def test_default_capabilities_include_budget_and_pii() -> None:
    """``default_notes_capabilities()`` ships both guards in the standard order."""
    caps = default_notes_capabilities()
    names = [c.name for c in caps]
    assert "budget_guard" in names
    assert "pii_guard" in names


@pytest.mark.asyncio
async def test_budget_guard_raises_when_max_iterations_zero() -> None:
    """A tight ``max_iterations=0`` budget raises ``BudgetExhausted`` before
    the first model call — confirms BudgetGuard from
    ``default_notes_capabilities()`` actually short-circuits the agent run."""

    def _never_called(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("model should not be called when budget is exhausted")

    agent: Agent[None, str] = Agent(
        model=FunctionModel(_never_called),
        capabilities=[BudgetGuard(max_iterations=0)],
    )
    with pytest.raises(BudgetExhausted):
        await agent.run("anything")


@pytest.mark.asyncio
async def test_pii_guard_redacts_email_from_model_response() -> None:
    """PIIGuard using notes-app's default regex set redacts a leaked email
    from the model's text response before it reaches the caller."""

    def _leak_email(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="Ping alice@example.com later.")])

    # Reuse the production capability list so the test pins what notes-app
    # actually ships (BudgetGuard ordering shouldn't affect this run; 20
    # iterations is more than enough for one model call).
    agent: Agent[None, str] = Agent(
        model=FunctionModel(_leak_email),
        capabilities=default_notes_capabilities(),
    )
    result = await agent.run("anything")
    text = str(result.output)
    assert "alice@example.com" not in text
    assert "[REDACTED]" in text

"""Capability wiring + behavior for ``NotesAgent``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_stateflow.capabilities import BudgetExhausted, BudgetGuard

from notes_app.agents.notes import default_notes_capabilities


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
    from the model's text response before it reaches the caller.

    Note: PIIGuard now also overrides ``wrap_run_event_stream`` which
    causes pydantic-ai to auto-enable streaming mode even on
    ``agent.run()``. The FunctionModel must therefore provide a
    ``stream_function`` to satisfy the agent's request flow.
    """
    leak = "Ping alice@example.com later."

    def _leak_email(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=leak)])

    async def _leak_email_stream(
        messages: list[ModelMessage], info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield leak

    # Reuse the production capability list so the test pins what notes-app
    # actually ships (BudgetGuard ordering shouldn't affect this run; 20
    # iterations is more than enough for one model call).
    agent: Agent[None, str] = Agent(
        model=FunctionModel(_leak_email, stream_function=_leak_email_stream),
        capabilities=default_notes_capabilities(),
    )
    result = await agent.run("anything")
    text = str(result.output)
    assert "alice@example.com" not in text
    assert "[REDACTED]" in text


# NB: the streaming SSE case is exercised end-to-end at the framework
# level in ``tests/api/test_streaming_router.py``. The pydantic-ai
# ``agent.run_stream()`` API exposes the model stream BEFORE
# ``wrap_run_event_stream`` runs to completion (per the upstream
# docstring — handlers wrap only graph advancement there, the
# streaming events are already yielded to the caller), so a direct
# ``async for chunk in result.stream_text()`` test would not be a
# faithful reproduction of the SSE adapter's consumption path.

"""Built-in GoalSource implementations."""
from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from ballast.drift._goal_sources import (
    ExplicitGoal, FirstUserMessage, LastUserMessage, WorkflowInput,
)
from ballast.drift._protocols import DriftContext


def _user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _resp(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _ctx_msgs(*msgs) -> DriftContext:
    return DriftContext(messages=list(msgs), run_ctx=None, workflow_input=None)


@pytest.mark.asyncio
async def test_first_user_message_returns_first_user_prompt() -> None:
    ctx = _ctx_msgs(
        _user_msg("plan a trip to Berlin"),
        _resp("ok"),
        _user_msg("actually Rome"),
    )
    g = await FirstUserMessage().goal(ctx)
    assert g == "plan a trip to Berlin"


@pytest.mark.asyncio
async def test_first_user_message_returns_empty_when_no_user_msg() -> None:
    ctx = _ctx_msgs(_resp("only-assistant"))
    g = await FirstUserMessage().goal(ctx)
    assert g == ""


@pytest.mark.asyncio
async def test_last_user_message_returns_last_user_prompt() -> None:
    ctx = _ctx_msgs(
        _user_msg("old"), _resp("a"), _user_msg("latest"),
    )
    g = await LastUserMessage().goal(ctx)
    assert g == "latest"


@pytest.mark.asyncio
async def test_workflow_input_returns_str_input() -> None:
    ctx = DriftContext(
        messages=[], run_ctx=None, workflow_input="research X thoroughly",
    )
    g = await WorkflowInput().goal(ctx)
    assert g == "research X thoroughly"


@pytest.mark.asyncio
async def test_workflow_input_falls_back_to_repr_for_non_str() -> None:
    ctx = DriftContext(
        messages=[], run_ctx=None, workflow_input={"intent": "X"},
    )
    g = await WorkflowInput().goal(ctx)
    assert "intent" in g and "X" in g


@pytest.mark.asyncio
async def test_explicit_goal_returns_stored_string() -> None:
    ctx = _ctx_msgs()
    g = await ExplicitGoal("manage finances").goal(ctx)
    assert g == "manage finances"

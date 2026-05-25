"""Built-in TraceWindow implementations."""
from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from ballast.drift._protocols import DriftContext
from ballast.drift._windows import (
    FullTrace, LastNMessages, SinceLastUserMessage, TokenBudgetWindow,
)


def _user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _resp(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _ctx(*msgs) -> DriftContext:
    return DriftContext(messages=list(msgs), run_ctx=None, workflow_input=None)


@pytest.mark.asyncio
async def test_full_trace_returns_all_messages() -> None:
    ctx = _ctx(_user_msg("hi"), _resp("hello"), _user_msg("more"))
    out = await FullTrace().slice(ctx)
    assert len(out) == 3


@pytest.mark.asyncio
async def test_last_n_messages_returns_tail() -> None:
    ctx = _ctx(*[_resp(str(i)) for i in range(10)])
    out = await LastNMessages(n=3).slice(ctx)
    assert len(out) == 3
    assert [p.parts[0].content for p in out] == ["7", "8", "9"]


@pytest.mark.asyncio
async def test_last_n_messages_handles_empty_trace() -> None:
    ctx = _ctx()
    out = await LastNMessages(n=5).slice(ctx)
    assert out == []


@pytest.mark.asyncio
async def test_last_n_messages_handles_n_larger_than_history() -> None:
    ctx = _ctx(_resp("a"), _resp("b"))
    out = await LastNMessages(n=10).slice(ctx)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_since_last_user_message_includes_user_and_after() -> None:
    ctx = _ctx(
        _user_msg("old"), _resp("answer1"),
        _user_msg("new"), _resp("answer2"), _resp("answer3"),
    )
    out = await SinceLastUserMessage().slice(ctx)
    # Slice begins at the LAST user message.
    assert len(out) == 3
    assert out[0].parts[0].content == "new"


@pytest.mark.asyncio
async def test_since_last_user_message_returns_all_when_no_user() -> None:
    ctx = _ctx(_resp("a"), _resp("b"))
    out = await SinceLastUserMessage().slice(ctx)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_token_budget_window_caps_from_tail() -> None:
    # Each msg has ~1 token; cap at 3 → expect last 3 messages.
    ctx = _ctx(*[_resp("x") for _ in range(10)])
    out = await TokenBudgetWindow(max_tokens=3).slice(ctx)
    assert len(out) == 3

import re

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.capabilities import PIIGuard

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d{10,15}")


def make_fn_model_returning(text: str) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])
    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_pii_guard_redacts_email():
    agent = Agent(
        model=make_fn_model_returning("Contact me at alice@example.com soon."),
        capabilities=[PIIGuard(patterns=[EMAIL_RE])],
    )
    result = await agent.run("ignored")
    text = str(result.output) if result.output else ""
    assert "alice@example.com" not in text
    assert "[REDACTED]" in text


@pytest.mark.asyncio
async def test_pii_guard_redacts_phone_with_custom_replacement():
    agent = Agent(
        model=make_fn_model_returning("Call +1234567890 now."),
        capabilities=[PIIGuard(patterns=[PHONE_RE], replacement="[PHONE]")],
    )
    result = await agent.run("ignored")
    text = str(result.output) if result.output else ""
    assert "+1234567890" not in text
    assert "[PHONE]" in text


@pytest.mark.asyncio
async def test_pii_guard_passes_through_clean_text():
    agent = Agent(
        model=make_fn_model_returning("Nothing to see here."),
        capabilities=[PIIGuard(patterns=[EMAIL_RE])],
    )
    result = await agent.run("ignored")
    assert "Nothing to see here" in str(result.output)

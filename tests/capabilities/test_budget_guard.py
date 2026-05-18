import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.capabilities import BudgetExhausted, BudgetGuard


def make_fn_model_returning(text: str, *, input_tokens: int = 10, output_tokens: int = 5) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # FunctionModel doesn't natively report usage; we use what's available.
        return ModelResponse(parts=[TextPart(content=text)])
    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_budget_guard_allows_run_within_iteration_limit():
    agent = Agent(model=make_fn_model_returning("ok"), capabilities=[BudgetGuard(max_iterations=10)])
    result = await agent.run("hi")
    assert "ok" in str(result.output).lower() or result.output == "ok"


@pytest.mark.asyncio
async def test_budget_guard_raises_when_max_iterations_zero():
    """A zero iteration budget refuses the first model call."""
    agent = Agent(model=make_fn_model_returning("ok"), capabilities=[BudgetGuard(max_iterations=0)])
    with pytest.raises(BudgetExhausted):
        await agent.run("hi")


def test_budget_guard_defaults_are_unlimited_for_tokens():
    """Without max_input_tokens / max_output_tokens, only iteration matters."""
    guard = BudgetGuard(max_iterations=5)
    assert guard.max_input_tokens is None
    assert guard.max_output_tokens is None
    assert guard.max_iterations == 5

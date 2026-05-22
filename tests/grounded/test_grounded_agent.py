from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ballast.grounded import GroundedAgent, Ref


class Item(BaseModel):
    id: UUID
    name: str


class Ctx(BaseModel):
    items: list[Item]


class Decision(BaseModel):
    chosen: Ref[Item]
    rationale: str


def make_function_model_returning_id(item_id: UUID) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={"chosen": str(item_id), "rationale": "always-first"},
        )])
    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_grounded_agent_run_returns_valid_decision():
    item_ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=item_ids[0], name="a"), Item(id=item_ids[1], name="b")])
    base_agent: Agent[None, Decision] = Agent(
        model=make_function_model_returning_id(item_ids[0]),
        output_type=Decision,
    )

    grounded = GroundedAgent(base_agent, output_type=Decision)
    result = await grounded.run(ctx, instructions="pick best")

    assert isinstance(result.value.chosen, Ref)
    assert result.value.chosen.id == item_ids[0]
    assert result.value.rationale == "always-first"


@pytest.mark.asyncio
async def test_grounded_agent_blocks_hallucinated_id():
    """If the function-model tries to return an id not in context, validation rejects."""
    item_ids = [uuid4()]
    ctx = Ctx(items=[Item(id=item_ids[0], name="a")])
    hallucinated = uuid4()
    base_agent: Agent[None, Decision] = Agent(
        model=make_function_model_returning_id(hallucinated),
        output_type=Decision,
    )

    grounded = GroundedAgent(base_agent, output_type=Decision)
    with pytest.raises(UnexpectedModelBehavior, match="retries"):
        await grounded.run(ctx, instructions="pick best")

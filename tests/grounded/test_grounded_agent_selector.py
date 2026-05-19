"""GroundedAgent honors ``Selector`` metadata on output fields.

When ``Annotated[Ref[T], Selector(...)]`` is present, the resolver calls
the selector at run-time and uses the returned IDs as the closed set —
overriding the HydrationMap-style context-scan default. Backward
compatible: bare ``Ref[T]`` fields still pick up IDs from the context.
"""

from typing import Annotated
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.grounded import GroundedAgent, Ref, Selector, SelectorRegistry


class _Item(BaseModel):
    id: UUID
    name: str


class _Ctx(BaseModel):
    items: list[_Item]


def _model_returning(item_id: UUID) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={"chosen": str(item_id), "rationale": "ok"},
        )])
    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_inline_selector_narrows_output_ids() -> None:
    """Selector returns only one of the two context items as valid."""
    allowed = uuid4()
    not_allowed = uuid4()
    ctx = _Ctx(items=[
        _Item(id=allowed, name="a"),
        _Item(id=not_allowed, name="b"),
    ])

    class _Decision(BaseModel):
        chosen: Annotated[Ref[_Item], Selector(lambda c: [allowed])]
        rationale: str

    base_agent: Agent[None, _Decision] = Agent(
        model=_model_returning(allowed),
        output_type=_Decision,
    )
    grounded = GroundedAgent(base_agent, output_type=_Decision)
    result = await grounded.run(ctx, instructions="pick")

    assert result.value.chosen.id == allowed


@pytest.mark.asyncio
async def test_named_selector_via_registry_on_output_field() -> None:
    allowed = uuid4()
    other = uuid4()
    ctx = _Ctx(items=[_Item(id=allowed, name="a"), _Item(id=other, name="b")])

    reg = SelectorRegistry()
    reg.register("open_items", lambda _c: [allowed])

    class _Decision(BaseModel):
        chosen: Annotated[Ref[_Item], Selector("open_items")]
        rationale: str

    base_agent: Agent[None, _Decision] = Agent(
        model=_model_returning(allowed),
        output_type=_Decision,
    )
    grounded = GroundedAgent(base_agent, output_type=_Decision, selectors=reg)
    result = await grounded.run(ctx, instructions="pick")

    assert result.value.chosen.id == allowed


@pytest.mark.asyncio
async def test_bare_ref_still_uses_context_scan() -> None:
    """No Selector → HydrationMap context-scan still works (backward-compat)."""
    item_id = uuid4()
    ctx = _Ctx(items=[_Item(id=item_id, name="a")])

    class _Decision(BaseModel):
        chosen: Ref[_Item]
        rationale: str

    base_agent: Agent[None, _Decision] = Agent(
        model=_model_returning(item_id),
        output_type=_Decision,
    )
    grounded = GroundedAgent(base_agent, output_type=_Decision)
    result = await grounded.run(ctx, instructions="pick")

    assert result.value.chosen.id == item_id

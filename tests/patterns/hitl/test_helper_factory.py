from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.patterns.hitl.helper.factory import (
    HelperAgentFactory,
    HelperDeps,
    HelperToolBox,
    make_helper_agent_with_approval_tools,
)
from pydantic_ai_stateflow.patterns.hitl.response import (
    ApprovedResponse,
    ModifiedResponse,
    RejectedResponse,
)
from pydantic_ai_stateflow.patterns.hitl.verdict import HelperVerdict


class _Ctx(BaseModel):
    note: str


_CtxVerdict = HelperVerdict[_Ctx]


def _scripted_agent(tool_name: str, tool_args: dict[str, Any]) -> Agent[HelperDeps, str]:
    """Build a pydantic-ai Agent that on its single turn calls `tool_name`."""

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # If a tool has already returned, emit a final text response.
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=tool_args)])

    return Agent[HelperDeps, str](model=FunctionModel(model_fn), deps_type=HelperDeps)


def _deps(request_id: UUID) -> HelperDeps:
    return HelperDeps(
        request_id=request_id, actor_id="founder",
        turn_count=0, tools_invoked_so_far=[], toolbox=HelperToolBox(),
        autopilot_eligible=False, cached_recommendation_confidence=None,
    )


def test_factory_protocol_satisfied_by_function():
    """The function `make_helper_agent_with_approval_tools` is a HelperAgentFactory."""
    assert callable(make_helper_agent_with_approval_tools)


@pytest.mark.asyncio
async def test_approve_tool_writes_response_and_verdict_to_toolbox():
    rid = uuid4()
    base = _scripted_agent("approve", {
        "rationale": "lgtm",
        "confidence": 0.9,
        "context": {"note": "all good"},
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    await agent.run("hi", deps=deps)
    assert deps.toolbox.response is not None
    assert isinstance(deps.toolbox.response, ApprovedResponse)
    assert deps.toolbox.response.feedback == "lgtm"
    assert deps.toolbox.response.helper_verdict is not None
    assert deps.toolbox.response.helper_verdict["rationale"] == "lgtm"
    assert deps.toolbox.response.helper_verdict["context"]["note"] == "all good"


@pytest.mark.asyncio
async def test_reject_tool_writes_rejected_response():
    rid = uuid4()
    base = _scripted_agent("reject", {
        "rationale": "missing evidence", "feedback": "cite sources",
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    await agent.run("hi", deps=deps)
    assert isinstance(deps.toolbox.response, RejectedResponse)
    assert deps.toolbox.response.feedback == "cite sources"
    assert deps.toolbox.response.helper_verdict is not None
    assert deps.toolbox.response.helper_verdict["rationale"] == "missing evidence"


@pytest.mark.asyncio
async def test_modify_tool_disabled_by_default():
    rid = uuid4()
    base = _scripted_agent("modify", {
        "rationale": "tweak",
        "confidence": 0.7,
        "modified_proposal": {"amount": 99},
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    with pytest.raises((UnexpectedModelBehavior, UsageLimitExceeded)):
        await agent.run("hi", deps=deps)
    assert deps.toolbox.response is None


@pytest.mark.asyncio
async def test_modify_tool_enabled_writes_modified_response():
    rid = uuid4()
    base = _scripted_agent("modify", {
        "rationale": "tweak",
        "confidence": 0.7,
        "modified_proposal": {"amount": 99},
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
        allow_modify=True,
    )
    deps = _deps(rid)
    await agent.run("hi", deps=deps)
    assert isinstance(deps.toolbox.response, ModifiedResponse)
    assert deps.toolbox.response.modified_proposal == {"amount": 99}


@pytest.mark.asyncio
async def test_finalize_partial_disabled_by_default():
    rid = uuid4()
    base = _scripted_agent("finalize_partial", {
        "rationale": "n/a", "approved_element_ids": [], "rejected_element_ids": [],
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    with pytest.raises((UnexpectedModelBehavior, UsageLimitExceeded)):
        await agent.run("hi", deps=deps)


def test_factory_returns_a_helper_agent_factory_protocol_value():
    rid = uuid4()
    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])
    base = Agent[HelperDeps, str](model=FunctionModel(model_fn), deps_type=HelperDeps)
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    assert isinstance(agent, Agent)


def test_helper_agent_factory_protocol_is_runtime_checkable():
    class _Mine:
        def __call__(
            self,
            *,
            base_agent: Agent[HelperDeps, str],
            request_id: UUID,
            context_type: type[Any] | None = None,
            allow_modify: bool = False,
            allow_partial: bool = False,
        ) -> Agent[HelperDeps, str]:
            return base_agent
    assert isinstance(_Mine(), HelperAgentFactory)

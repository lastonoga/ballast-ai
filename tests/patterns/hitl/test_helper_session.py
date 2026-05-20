from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
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
    HelperDeps,
    make_helper_agent_with_approval_tools,
)
from pydantic_ai_stateflow.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
    HelperSessionRunner,
)
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence import InMemoryThreadRepository


class _Ctx(BaseModel):
    note: str


def _scripted_agent(
    plan: list[tuple[str, dict[str, Any]]],
) -> Agent[HelperDeps, str]:
    state = {"i": 0}

    async def model_fn(
        messages: list[ModelMessage], info: AgentInfo,
    ) -> ModelResponse:
        for msg in reversed(messages):
            if isinstance(msg, ModelRequest):
                if any(isinstance(p, ToolReturnPart) for p in msg.parts):
                    return ModelResponse(parts=[TextPart(content="done")])
                break
        idx = state["i"]
        state["i"] += 1
        tool_name, tool_args = plan[idx]
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool_name, args=tool_args)],
        )

    return Agent[HelperDeps, str](
        model=FunctionModel(model_fn), deps_type=HelperDeps,
    )


def _build_runner(agent_factory) -> DefaultHelperSessionRunner:
    return DefaultHelperSessionRunner(
        thread_repo=InMemoryThreadRepository(),
        agent_factory=agent_factory,
        max_turns=5,
    )


def test_runner_satisfies_protocol():
    runner = _build_runner(
        lambda *, base_agent, request_id, context_type=None, **k: base_agent,
    )
    assert isinstance(runner, HelperSessionRunner)


@pytest.mark.asyncio
async def test_runner_sends_response_to_gate_topic_and_exits(
    fresh_dbos_executor,
):
    rid = uuid4()
    wf_id = uuid4()
    prompt = HITLPrompt(title="t", context="c", decision_kinds={"approved"})
    base = _scripted_agent([(
        "approve",
        {"rationale": "ok", "confidence": 0.9},
    )])

    def factory(*, base_agent, request_id, context_type=None, **k):
        return make_helper_agent_with_approval_tools(
            base_agent=base_agent,
            request_id=request_id,
            context_type=context_type,
        )

    runner = _build_runner(factory)
    sent: dict[str, Any] = {}

    recv = AsyncMock(side_effect=[{"text": "hello"}, None])

    def fake_send(destination_id, message, topic=None):
        sent.update(
            destination_id=destination_id, message=message, topic=topic,
        )

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.helper.session.DBOS.recv", recv,
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.helper.session.DBOS.send",
        fake_send,
    ):
        runner._base_agent_for_test = base
        await runner.run(
            HelperSessionInput(
                prompt_payload=prompt.model_dump(mode="json"),
                request_id=rid,
                gate_workflow_id=wf_id,
                base_agent_module="tests.patterns.hitl.test_helper_session",
                base_agent_attr=None,
                context_type_fqn=None,
                actor_id="founder",
            ),
        )

    assert sent["destination_id"] == str(wf_id)
    assert sent["topic"] == _hitl_topic(rid)
    assert sent["message"]["kind"] == "approved"
    assert len(runner.thread_repo._threads) == 1
    th = next(iter(runner.thread_repo._threads.values()))
    assert th.agent == "hitl"
    assert th.metadata_["request_id"] == str(rid)


@pytest.mark.asyncio
async def test_runner_loops_on_non_verdict_messages_then_completes(
    fresh_dbos_executor,
):
    rid = uuid4()
    wf_id = uuid4()
    prompt = HITLPrompt(title="t", context="c", decision_kinds={"approved"})

    base = _scripted_agent([
        ("reject", {"rationale": "need more info"}),
        ("approve", {
            "rationale": "ok", "confidence": 0.9, "context": {"note": "n"},
        }),
    ])

    def factory(*, base_agent, request_id, context_type=None, **k):
        return make_helper_agent_with_approval_tools(
            base_agent=base_agent,
            request_id=request_id,
            context_type=context_type,
        )

    runner = _build_runner(factory)
    recv = AsyncMock(side_effect=[
        {"text": "first"},
        {"text": "second"},
    ])
    sends: list[dict[str, Any]] = []

    def fake_send(destination_id, message, topic=None):
        sends.append({
            "destination_id": destination_id,
            "topic": topic,
            "message": message,
        })

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.helper.session.DBOS.recv", recv,
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.helper.session.DBOS.send",
        fake_send,
    ):
        runner._base_agent_for_test = base
        await runner.run(
            HelperSessionInput(
                prompt_payload=prompt.model_dump(mode="json"),
                request_id=rid,
                gate_workflow_id=wf_id,
                base_agent_module="x",
                base_agent_attr=None,
                context_type_fqn=None,
                actor_id="founder",
            ),
        )

    assert len(sends) == 1
    assert sends[0]["message"]["kind"] == "rejected"
    assert recv.await_count == 1


@pytest.mark.asyncio
async def test_runner_bounded_by_max_turns(fresh_dbos_executor):
    """If max_turns exhausted with no verdict, runner exits without sending."""
    rid = uuid4()
    wf_id = uuid4()
    prompt = HITLPrompt(title="t", context="c", decision_kinds={"approved"})

    async def model_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="thinking...")])

    base = Agent[HelperDeps, str](
        model=FunctionModel(model_fn), deps_type=HelperDeps,
    )

    def factory(*, base_agent, request_id, context_type=None, **k):
        return make_helper_agent_with_approval_tools(
            base_agent=base_agent,
            request_id=request_id,
            context_type=context_type,
        )

    runner = DefaultHelperSessionRunner(
        thread_repo=InMemoryThreadRepository(),
        agent_factory=factory,
        max_turns=3,
    )
    recv = AsyncMock(return_value={"text": "noop"})
    sends: list[Any] = []

    def fake_send(*a, **k):
        sends.append((a, k))

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.helper.session.DBOS.recv", recv,
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.helper.session.DBOS.send",
        fake_send,
    ):
        runner._base_agent_for_test = base
        await runner.run(
            HelperSessionInput(
                prompt_payload=prompt.model_dump(mode="json"),
                request_id=rid,
                gate_workflow_id=wf_id,
                base_agent_module="x",
                base_agent_attr=None,
                context_type_fqn=None,
                actor_id="founder",
            ),
        )

    assert recv.await_count == 3
    assert sends == []


def test_helper_session_input_is_frozen_basemodel():
    rid = uuid4()
    wf = uuid4()
    inp = HelperSessionInput(
        prompt_payload={"k": "v"},
        request_id=rid,
        gate_workflow_id=wf,
        base_agent_module="m",
        base_agent_attr=None,
        context_type_fqn=None,
        actor_id="a",
    )
    with pytest.raises(ValidationError):
        inp.request_id = uuid4()  # type: ignore[misc]

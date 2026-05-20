from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.patterns.hitl.channel import HITLChannel
from pydantic_ai_stateflow.patterns.hitl.channels.conversational import (
    ConversationalChannel,
)
from pydantic_ai_stateflow.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
)
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    ApprovedResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence import InMemoryThreadRepository


class _Ctx(BaseModel):
    note: str


def _make_channel(runner: Any = None) -> ConversationalChannel:
    runner = runner or MagicMock(spec=DefaultHelperSessionRunner)
    return ConversationalChannel(
        helper_session_runner=runner,
        thread_repo=InMemoryThreadRepository(),
        base_agent_module="tests.patterns.hitl.test_conversational_channel",
        base_agent_attr=None,
        context_type=_Ctx,
        gate_workflow_id_resolver=lambda: uuid4(),
    )


def test_conversational_channel_satisfies_protocol() -> None:
    assert isinstance(_make_channel(), HITLChannel)


@pytest.mark.asyncio
async def test_ask_starts_helper_session_then_recvs(fresh_dbos_executor: Any) -> None:
    rid = uuid4()
    gate_wf = uuid4()
    prompt = HITLPrompt(
        title="strategy review",
        context="c",
        decision_kinds={"approved"},
        timeout=timedelta(seconds=5),
    )

    started_with: dict[str, Any] = {}

    async def fake_start(
        workflow: Any, input: Any, *, idempotency_key: str | None = None,
    ) -> None:
        started_with["workflow"] = workflow
        started_with["input"] = input
        started_with["idempotency_key"] = idempotency_key
        return

    runner = MagicMock()
    runner.run = MagicMock()
    payload = ApprovedResponse(
        actor_id="founder", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)

    channel = ConversationalChannel(
        helper_session_runner=runner,
        thread_repo=InMemoryThreadRepository(),
        base_agent_module="my_app.agents",
        base_agent_attr="strategy_helper",
        context_type=_Ctx,
        gate_workflow_id_resolver=lambda: gate_wf,
    )

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational"
        ".start_workflow_async", fake_start,
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational.DBOS.recv",
        recv,
    ):
        result = await channel.ask(prompt, request_id=rid)

    assert isinstance(result, ApprovedResponse)
    assert started_with["workflow"] is runner.run
    inp: HelperSessionInput = started_with["input"]
    assert inp.request_id == rid
    assert inp.gate_workflow_id == gate_wf
    assert inp.base_agent_module == "my_app.agents"
    assert inp.base_agent_attr == "strategy_helper"
    assert inp.context_type_fqn is not None
    assert inp.context_type_fqn.endswith("test_conversational_channel._Ctx")
    assert started_with["idempotency_key"].startswith("helper:")
    recv.assert_awaited_once_with(_hitl_topic(rid), timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_ask_returns_timeout_when_recv_returns_none(
    fresh_dbos_executor: Any,
) -> None:
    rid = uuid4()
    prompt = HITLPrompt(
        title="t", context="c", decision_kinds={"approved"},
        timeout=timedelta(seconds=1),
    )
    runner = MagicMock()
    runner.run = MagicMock()
    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational"
        ".start_workflow_async", AsyncMock(return_value=None),
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational.DBOS.recv",
        AsyncMock(return_value=None),
    ):
        channel = ConversationalChannel(
            helper_session_runner=runner,
            thread_repo=InMemoryThreadRepository(),
            base_agent_module="m", base_agent_attr=None,
            context_type=None,
            gate_workflow_id_resolver=lambda: uuid4(),
        )
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, TimeoutResponse)


@pytest.mark.asyncio
async def test_idempotency_key_stable_for_same_request(
    fresh_dbos_executor: Any,
) -> None:
    rid = uuid4()
    prompt = HITLPrompt(title="t", context="c", decision_kinds={"approved"})
    keys: list[str] = []

    async def fake_start(
        workflow: Any, input: Any, *, idempotency_key: str | None = None,
    ) -> None:
        assert idempotency_key is not None
        keys.append(idempotency_key)
        return

    runner = MagicMock()
    payload = ApprovedResponse(
        actor_id="f", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational"
        ".start_workflow_async", fake_start,
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational.DBOS.recv",
        AsyncMock(return_value=payload),
    ):
        channel = ConversationalChannel(
            helper_session_runner=runner,
            thread_repo=InMemoryThreadRepository(),
            base_agent_module="m", base_agent_attr=None,
            context_type=None,
            gate_workflow_id_resolver=lambda: uuid4(),
        )
        await channel.ask(prompt, request_id=rid)
        await channel.ask(prompt, request_id=rid)
    assert len(keys) == 2
    assert keys[0] == keys[1]

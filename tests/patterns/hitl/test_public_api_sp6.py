from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pydantic_ai_stateflow import (
    ConversationalChannel,
    DefaultHelperSessionRunner,
    HelperVerdict,
    HITLGate,
    HITLPrompt,
    UIChannel,
    WebhookChannel,
    WebhookConfig,
    build_hitl_router,
    make_helper_agent_with_approval_tools,
)
from pydantic_ai_stateflow.patterns.hitl.policy import AllowAll
from pydantic_ai_stateflow.patterns.hitl.response import ApprovedResponse
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence import (
    InMemoryHITLRepository,
    InMemoryThreadRepository,
)


class _Ctx(BaseModel):
    note: str


_VerdictSmoke = HelperVerdict[_Ctx]


def test_all_sp6_symbols_visible_at_top_level() -> None:
    assert UIChannel is not None
    assert WebhookChannel is not None
    assert WebhookConfig is not None
    assert ConversationalChannel is not None
    assert HelperVerdict is not None
    assert make_helper_agent_with_approval_tools is not None
    assert build_hitl_router is not None


def test_ui_channel_router_dispatches_to_gate_topic(
    fresh_dbos_executor: None,
) -> None:
    """Smoke: POSTing to /hitl/{rid}/respond fires DBOS.send to the right topic."""
    wf_id = uuid4()
    repo = InMemoryHITLRepository()
    req = asyncio.run(
        repo.persist_request(
            prompt={
                "title": "x",
                "context": "y",
                "decision_kinds": ["approved"],
            },
            workflow_id=wf_id,
            gate_kind="hitl_gate",
            purpose="approval",
        )
    )
    app = FastAPI()
    app.include_router(build_hitl_router(repo=repo, policy=AllowAll()))

    sent: dict[str, object] = {}

    def fake_send(destination_id, message, topic=None):  # type: ignore[no-untyped-def]
        sent.update(destination_id=destination_id, message=message, topic=topic)

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.api.router.DBOS.send", fake_send,
    ), TestClient(app) as client:
        r = client.post(
            f"/hitl/{req.id}/respond",
            json={
                "kind": "approved",
                "actor_id": "alice",
                "answered_at": datetime.now(tz=UTC).isoformat(),
            },
        )

    assert r.status_code == 200
    assert sent["topic"] == _hitl_topic(req.id)
    assert sent["destination_id"] == str(wf_id)


@pytest.mark.asyncio
async def test_conversational_channel_exposes_helper_verdict_to_persistence(
    fresh_dbos_executor: None,
) -> None:
    """SP6 acceptance: helper verdict from ConversationalChannel's helper
    agent reaches the Decision row via HITLGate's persistence wiring (Task 9)."""
    repo = InMemoryHITLRepository()
    runner = MagicMock(spec=DefaultHelperSessionRunner)
    runner.run = MagicMock()
    channel = ConversationalChannel(
        helper_session_runner=runner,
        thread_repo=InMemoryThreadRepository(),
        base_agent_module="x",
        base_agent_attr=None,
        context_type=_Ctx,
        gate_workflow_id_resolver=lambda: uuid4(),
    )
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)
    verdict_dict = _VerdictSmoke(
        rationale="r",
        confidence=0.9,
        conversation_turn_count=2,
        tools_invoked=["approve"],
        context=_Ctx(note="ok"),
    ).model_dump(mode="json")

    async def fake_start(workflow, input, *, idempotency_key):  # type: ignore[no-untyped-def]
        return None

    payload = ApprovedResponse(
        actor_id="founder",
        answered_at=datetime.now(tz=UTC),
        helper_verdict=verdict_dict,
    ).model_dump(mode="json")

    prompt = HITLPrompt(title="t", context="c", decision_kinds={"approved"})
    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational.start_workflow_async",
        fake_start,
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.conversational.DBOS.recv",
        AsyncMock(return_value=payload),
    ):
        await gate.run(prompt)

    decision = next(iter(repo._decisions.values()))
    assert decision.helper_verdict_payload is not None
    assert decision.helper_verdict_payload["rationale"] == "r"

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.patterns.hitl.channel import InMemoryHITLChannel
from pydantic_ai_stateflow.patterns.hitl.gate import HITLGate
from pydantic_ai_stateflow.patterns.hitl.policy import AllowAll
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import ApprovedResponse
from pydantic_ai_stateflow.patterns.hitl.verdict import HelperVerdict
from pydantic_ai_stateflow.persistence import InMemoryHITLRepository


class _Ctx(BaseModel):
    note: str


_CtxVerdict_Wiring = HelperVerdict[_Ctx]


@pytest.mark.asyncio
async def test_helper_verdict_persisted_via_gate(fresh_dbos_executor: None) -> None:
    tid = uuid4()
    repo = InMemoryHITLRepository()
    channel = InMemoryHITLChannel()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    verdict = _CtxVerdict_Wiring(
        rationale="r",
        confidence=0.9,
        conversation_turn_count=2,
        tools_invoked=["approve"],
        context=_Ctx(note="hello"),
    )

    captured: dict[str, object] = {}
    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        captured["request"] = req
        channel.set_response(
            req.id,
            ApprovedResponse(
                actor_id="alice",
                answered_at=datetime.now(tz=UTC),
                feedback="ok",
                helper_verdict=verdict.model_dump(mode="json"),
            ),
        )
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    prompt = HITLPrompt(
        title="t", context="c", decision_kinds={"approved"},
    )
    await gate.run(prompt)

    assert len(repo._decisions) == 1
    decision = next(iter(repo._decisions.values()))
    assert decision.helper_verdict_payload is not None
    assert decision.helper_verdict_payload["rationale"] == "r"
    assert decision.helper_verdict_payload["context"]["note"] == "hello"


@pytest.mark.asyncio
async def test_helper_verdict_absent_when_response_lacks_it(
    fresh_dbos_executor: None,
) -> None:
    tid = uuid4()
    repo = InMemoryHITLRepository()
    channel = InMemoryHITLChannel()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        channel.set_response(
            req.id,
            ApprovedResponse(actor_id="alice", answered_at=datetime.now(tz=UTC)),
        )
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    prompt = HITLPrompt(
        title="t", context="c", decision_kinds={"approved"},
    )
    await gate.run(prompt)
    decision = next(iter(repo._decisions.values()))
    assert decision.helper_verdict_payload is None
    assert decision.helper_verdict_context_type is None
    assert decision.helper_thread_id is None


@pytest.mark.asyncio
async def test_helper_thread_id_propagated_via_prompt_metadata(
    fresh_dbos_executor: None,
) -> None:
    """If the helper_verdict carries the sidecar keys (set by ConversationalChannel
    via the helper agent), the gate threads them into the typed Decision columns
    and strips them from the persisted payload."""
    tid = uuid4()
    thread_id = uuid4()
    repo = InMemoryHITLRepository()
    channel = InMemoryHITLChannel()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        channel.set_response(
            req.id,
            ApprovedResponse(
                actor_id="alice",
                answered_at=datetime.now(tz=UTC),
                helper_verdict={
                    "rationale": "r",
                    "confidence": 1.0,
                    "conversation_turn_count": 0,
                    "tools_invoked": [],
                    "autopilot_eligible": False,
                    "autopilot_confidence": None,
                    "context": None,
                    "__helper_thread_id__": str(thread_id),
                    "__context_type_fqn__": "x.y.Ctx",
                },
            ),
        )
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    prompt = HITLPrompt(
        title="t", context="c", decision_kinds={"approved"},
    )
    await gate.run(prompt)
    decision = next(iter(repo._decisions.values()))
    assert decision.helper_thread_id == thread_id
    assert decision.helper_verdict_context_type == "x.y.Ctx"
    # sidecar keys stripped from persisted payload
    assert decision.helper_verdict_payload is not None
    assert "__helper_thread_id__" not in decision.helper_verdict_payload
    assert "__context_type_fqn__" not in decision.helper_verdict_payload

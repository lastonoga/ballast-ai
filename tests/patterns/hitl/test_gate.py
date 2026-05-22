from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.patterns import HITLDenied, HITLGate, HITLTimedOut
from ballast.patterns.hitl import (
    AllowAll,
    ApprovedResponse,
    DenyAll,
    HITLPrompt,
    InMemoryHITLChannel,
    RejectedResponse,
    TimeoutResponse,
)
from ballast.persistence import InMemoryHITLRepository


@pytest.mark.asyncio
async def test_gate_returns_approved_response_when_policy_allows(
    fresh_dbos_executor: None,
) -> None:
    channel = InMemoryHITLChannel()
    repo = InMemoryHITLRepository()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    prompt = HITLPrompt(
        title="Approve refund", context="$50",
        decision_kinds={"approved", "rejected"},
    )
    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        channel.set_response(
            req.id,
            ApprovedResponse(actor_id="alice", answered_at=datetime.now(tz=UTC)),
        )
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    resp = await gate.run(prompt)
    assert isinstance(resp, ApprovedResponse)
    assert resp.actor_id == "alice"


@pytest.mark.asyncio
async def test_gate_raises_hitl_denied_when_policy_rejects_responder(
    fresh_dbos_executor: None,
) -> None:
    channel = InMemoryHITLChannel()
    repo = InMemoryHITLRepository()
    gate = HITLGate(channel=channel, policy=DenyAll(), repo=repo)

    prompt = HITLPrompt(
        title="x", context="y", decision_kinds={"approved"},
    )
    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        channel.set_response(
            req.id,
            ApprovedResponse(actor_id="mallory", answered_at=datetime.now(tz=UTC)),
        )
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    with pytest.raises(HITLDenied) as exc:
        await gate.run(prompt)
    assert exc.value.actor_id == "mallory"


@pytest.mark.asyncio
async def test_gate_persists_authz_denial_audit_row(
    fresh_dbos_executor: None,
) -> None:
    channel = InMemoryHITLChannel()
    repo = InMemoryHITLRepository()
    gate = HITLGate(channel=channel, policy=DenyAll(), repo=repo)

    prompt = HITLPrompt(
        title="x", context="y", decision_kinds={"approved"},
    )
    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        channel.set_response(
            req.id,
            ApprovedResponse(actor_id="mallory", answered_at=datetime.now(tz=UTC)),
        )
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    with pytest.raises(HITLDenied):
        await gate.run(prompt)

    assert len(repo._denials) == 1
    assert repo._denials[0].actor_id == "mallory"


@pytest.mark.asyncio
async def test_gate_raises_timeout_when_channel_returns_timeout(
    fresh_dbos_executor: None,
) -> None:
    channel = InMemoryHITLChannel()
    repo = InMemoryHITLRepository()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    prompt = HITLPrompt(
        title="x", context="y", decision_kinds={"approved"},
    )
    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        channel.set_response(req.id, TimeoutResponse(answered_at=datetime.now(tz=UTC)))
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    with pytest.raises(HITLTimedOut):
        await gate.run(prompt)


@pytest.mark.asyncio
async def test_gate_persists_response_on_grant(
    fresh_dbos_executor: None,
) -> None:
    channel = InMemoryHITLChannel()
    repo = InMemoryHITLRepository()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    prompt = HITLPrompt(
        title="x", context="y", decision_kinds={"rejected"},
    )
    orig = repo.persist_request

    async def capture(**kw):  # type: ignore[no-untyped-def]
        req = await orig(**kw)
        channel.set_response(
            req.id,
            RejectedResponse(
                actor_id="alice",
                answered_at=datetime.now(tz=UTC),
                feedback="no",
            ),
        )
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    resp = await gate.run(prompt)
    assert isinstance(resp, RejectedResponse)
    assert len(repo._decisions) == 1
    assert next(iter(repo._decisions.values())).actor_id == "alice"

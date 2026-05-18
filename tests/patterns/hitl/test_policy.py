from __future__ import annotations

from uuid import uuid4

import pytest

from pydantic_ai_stateflow.patterns.hitl import (
    AllowAll,
    DenyAll,
    Policy,
)


def test_allow_all_satisfies_policy_protocol() -> None:
    assert isinstance(AllowAll(), Policy)


def test_deny_all_satisfies_policy_protocol() -> None:
    assert isinstance(DenyAll(), Policy)


@pytest.mark.asyncio
async def test_allow_all_grants_any_actor() -> None:
    p = AllowAll()
    verdict = await p.can(
        actor="alice", action="decide", resource={"x": 1}, tenant_id=uuid4(),
    )
    assert verdict.is_grant is True


@pytest.mark.asyncio
async def test_deny_all_denies_any_actor() -> None:
    p = DenyAll()
    verdict = await p.can(
        actor="alice", action="decide", resource={"x": 1}, tenant_id=uuid4(),
    )
    assert verdict.is_grant is False
    assert "deny" in verdict.votes.get("deny_all", "")

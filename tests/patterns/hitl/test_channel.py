from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.patterns.hitl import (
    ApprovedResponse,
    HITLChannel,
    HITLPrompt,
    InMemoryHITLChannel,
)


def test_in_memory_channel_satisfies_protocol() -> None:
    assert isinstance(InMemoryHITLChannel(), HITLChannel)


@pytest.mark.asyncio
async def test_in_memory_channel_returns_preloaded_response() -> None:
    channel = InMemoryHITLChannel()
    request_id = uuid4()
    prompt = HITLPrompt(
        tenant_id=uuid4(), title="x", context="y", decision_kinds={"approved"},
    )
    expected = ApprovedResponse(actor_id="alice", answered_at=datetime.now(tz=UTC))
    channel.set_response(request_id, expected)
    got = await channel.ask(prompt, request_id=request_id)
    assert got is expected


@pytest.mark.asyncio
async def test_in_memory_channel_raises_if_no_response_set() -> None:
    channel = InMemoryHITLChannel()
    prompt = HITLPrompt(
        tenant_id=uuid4(), title="x", context="y", decision_kinds={"approved"},
    )
    with pytest.raises(KeyError):
        await channel.ask(prompt, request_id=uuid4())

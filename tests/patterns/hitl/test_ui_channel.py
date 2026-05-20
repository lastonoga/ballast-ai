from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.patterns.hitl.channel import HITLChannel
from pydantic_ai_stateflow.patterns.hitl.channels.ui import UIChannel
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    ApprovedResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic


def test_ui_channel_satisfies_protocol():
    assert isinstance(UIChannel(), HITLChannel)


@pytest.mark.asyncio
async def test_ui_channel_returns_received_response():
    rid = uuid4()
    prompt = HITLPrompt(
        title="t",
        context="c",
        decision_kinds={"approved", "rejected"},
        timeout=timedelta(seconds=5),
    )
    payload = ApprovedResponse(
        actor_id="alice", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)
    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.ui.DBOS.recv_async", recv,
    ):
        channel = UIChannel()
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, ApprovedResponse)
    assert result.actor_id == "alice"
    recv.assert_awaited_once_with(_hitl_topic(rid), timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_ui_channel_returns_timeout_on_none():
    rid = uuid4()
    prompt = HITLPrompt(
        title="t",
        context="c",
        decision_kinds={"approved"},
        timeout=timedelta(seconds=1),
    )
    recv = AsyncMock(return_value=None)
    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.ui.DBOS.recv_async", recv,
    ):
        channel = UIChannel()
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, TimeoutResponse)


@pytest.mark.asyncio
async def test_ui_channel_no_timeout_passes_large_finite_ceiling():
    """No ``prompt.timeout`` ⇒ effectively-infinite finite timeout.

    ``DBOS.recv_async`` does ``time.time() + seconds`` internally, so
    ``None`` would crash. UIChannel substitutes ``_NO_TIMEOUT_SECONDS``
    (≈ 1 year) instead.
    """
    rid = uuid4()
    prompt = HITLPrompt(
        title="t",
        context="c",
        decision_kinds={"approved"},
    )
    payload = ApprovedResponse(
        actor_id="bob", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)
    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.ui.DBOS.recv_async", recv,
    ):
        channel = UIChannel()
        await channel.ask(prompt, request_id=rid)
    recv.assert_awaited_once_with(
        _hitl_topic(rid), timeout_seconds=UIChannel._NO_TIMEOUT_SECONDS,
    )

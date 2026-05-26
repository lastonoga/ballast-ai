"""ApprovalCapability — bridges pydantic-ai requires_approval to UICardChannel."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

from ballast.capabilities.approval import ApprovalCapability
from ballast.patterns.hitl import CardVerdict


_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _PublishPayload(BaseModel):
    __hitl_kind__: ClassVar[str] = "test-publish"
    title: str
    body: str


class _PublishVerdict(CardVerdict[_PublishPayload]):
    __hitl_kind__: ClassVar[str] = "test-publish"


@dataclass
class _FakeAgentResult:
    output: Any
    _messages: list = None

    def all_messages(self) -> list:
        return self._messages or []


class _FakeChannel:
    def __init__(self, verdicts: list[CardVerdict]):
        self.verdicts = list(verdicts)
        self.requests: list = []

    async def request(self, payload, *, timeout=None):
        self.requests.append({"payload": payload, "timeout": timeout})
        return self.verdicts.pop(0)


@pytest.mark.asyncio
async def test_wrap_run_passes_through_when_no_deferred_requests() -> None:
    cap = ApprovalCapability(tool_card_map={})
    final_result = _FakeAgentResult(output="normal text response")
    handler = AsyncMock(return_value=final_result)
    handler.__self__ = AsyncMock()

    fake_ctx = AsyncMock()
    out = await cap.wrap_run(fake_ctx, handler=handler)
    assert out is final_result
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_wrap_run_opens_card_for_each_deferred_approval() -> None:
    captured_payloads = []

    def make_payload(tc: ToolCallPart, deps: Any) -> _PublishPayload:
        captured_payloads.append(tc)
        args = tc.args if isinstance(tc.args, dict) else {}
        return _PublishPayload(title=args.get("title", ""), body=args.get("body", ""))

    channel = _FakeChannel([
        _PublishVerdict(decision="approve", modified=None, feedback=None, answered_at=_NOW),
    ])

    cap = ApprovalCapability(
        tool_card_map={
            "publish_note": (_PublishPayload, make_payload, channel),
        },
    )

    deferred = DeferredToolRequests(
        calls=[],
        approvals=[
            ToolCallPart(tool_name="publish_note", args={"title": "T", "body": "B"}, tool_call_id="tc-1"),
        ],
    )
    first_result = _FakeAgentResult(output=deferred, _messages=[])
    final_result = _FakeAgentResult(output="done after approval")

    handler = AsyncMock(return_value=first_result)
    fake_agent = AsyncMock()
    fake_agent.run = AsyncMock(return_value=final_result)
    handler.__self__ = fake_agent

    fake_ctx = AsyncMock()
    fake_ctx.deps = None

    out = await cap.wrap_run(fake_ctx, handler=handler)

    assert out is final_result
    assert len(channel.requests) == 1
    assert isinstance(channel.requests[0]["payload"], _PublishPayload)
    assert channel.requests[0]["payload"].title == "T"

    fake_agent.run.assert_awaited_once()
    call_kwargs = fake_agent.run.await_args.kwargs
    deferred_results: DeferredToolResults = call_kwargs["deferred_tool_results"]
    approvals = deferred_results.approvals
    assert "tc-1" in approvals
    assert isinstance(approvals["tc-1"], ToolApproved)


@pytest.mark.asyncio
async def test_wrap_run_maps_reject_to_tool_denied() -> None:
    channel = _FakeChannel([
        _PublishVerdict(
            decision="reject", modified=None,
            feedback="not now", answered_at=_NOW,
        ),
    ])

    cap = ApprovalCapability(
        tool_card_map={
            "publish_note": (
                _PublishPayload,
                lambda tc, deps: _PublishPayload(title="x", body="y"),
                channel,
            ),
        },
    )

    deferred = DeferredToolRequests(
        calls=[],
        approvals=[
            ToolCallPart(tool_name="publish_note", args={}, tool_call_id="tc-1"),
        ],
    )
    first_result = _FakeAgentResult(output=deferred, _messages=[])
    final_result = _FakeAgentResult(output="done after rejection")

    handler = AsyncMock(return_value=first_result)
    fake_agent = AsyncMock()
    fake_agent.run = AsyncMock(return_value=final_result)
    handler.__self__ = fake_agent
    fake_ctx = AsyncMock()
    fake_ctx.deps = None

    await cap.wrap_run(fake_ctx, handler=handler)

    deferred_results = fake_agent.run.await_args.kwargs["deferred_tool_results"]
    denied = deferred_results.approvals["tc-1"]
    assert isinstance(denied, ToolDenied)
    assert "not now" in denied.message


@pytest.mark.asyncio
async def test_wrap_run_maps_modified_to_override_args() -> None:
    modified_payload = _PublishPayload(title="EDITED", body="EDITED-BODY")
    channel = _FakeChannel([
        _PublishVerdict(
            decision="approve", modified=modified_payload,
            feedback=None, answered_at=_NOW,
        ),
    ])

    cap = ApprovalCapability(
        tool_card_map={
            "publish_note": (
                _PublishPayload,
                lambda tc, deps: _PublishPayload(title="x", body="y"),
                channel,
            ),
        },
    )

    deferred = DeferredToolRequests(
        calls=[],
        approvals=[ToolCallPart(tool_name="publish_note", args={}, tool_call_id="tc-1")],
    )
    handler = AsyncMock(return_value=_FakeAgentResult(output=deferred, _messages=[]))
    fake_agent = AsyncMock()
    fake_agent.run = AsyncMock(return_value=_FakeAgentResult(output="ok"))
    handler.__self__ = fake_agent
    fake_ctx = AsyncMock()
    fake_ctx.deps = None

    await cap.wrap_run(fake_ctx, handler=handler)

    deferred_results = fake_agent.run.await_args.kwargs["deferred_tool_results"]
    approved = deferred_results.approvals["tc-1"]
    assert isinstance(approved, ToolApproved)
    assert approved.override_args == {"title": "EDITED", "body": "EDITED-BODY"}


@pytest.mark.asyncio
async def test_wrap_run_denies_unmapped_tool_with_helpful_message() -> None:
    cap = ApprovalCapability(tool_card_map={})

    deferred = DeferredToolRequests(
        calls=[],
        approvals=[ToolCallPart(tool_name="unmapped_tool", args={}, tool_call_id="tc-1")],
    )
    first_result = _FakeAgentResult(output=deferred, _messages=[])
    handler = AsyncMock(return_value=first_result)
    fake_agent = AsyncMock()
    fake_agent.run = AsyncMock(return_value=_FakeAgentResult(output="ok"))
    handler.__self__ = fake_agent
    fake_ctx = AsyncMock()
    fake_ctx.deps = None

    await cap.wrap_run(fake_ctx, handler=handler)

    deferred_results = fake_agent.run.await_args.kwargs["deferred_tool_results"]
    denied = deferred_results.approvals["tc-1"]
    assert isinstance(denied, ToolDenied)
    assert "tool_card_map" in denied.message or "registered" in denied.message

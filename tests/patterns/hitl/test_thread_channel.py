"""``ThreadChannel`` — helper-thread HITL channel.

Test plan
---------
1. Constructor guards: ValueError when helper_agent.metadata_model is None;
   TypeError when metadata_model != payload_type.
2. Happy path: deliver() calls _create_helper_thread + emits
   helper_thread_created signal; workflow waits then returns CardVerdict.
3. Timeout: no send_async within tight deadline → TimeoutError.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from ballast.durable import Durable
from ballast.events import helper_thread_created
from ballast.patterns.hitl.channels.thread import ThreadChannel
from ballast.patterns.hitl.channels.ui_card import CardVerdict
from ballast.runtime.agents import BallastAgent


# ── fixture models ───────────────────────────────────────────────────────────

class _TodoCtx(BaseModel):
    proposed_title: str
    proposed_body: str
    parent_thread_id: str


class _FakeHelperAgent(BallastAgent):
    name = "fake-helper-agent"
    metadata_model = _TodoCtx

    def build_agent(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    def build_deps(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _AgentWithoutMetadataModel(BallastAgent):
    name = "no-meta-agent"
    metadata_model = None

    def build_agent(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    def build_deps(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _AgentWithWrongMetadataModel(BallastAgent):
    name = "wrong-meta-agent"
    metadata_model = _TodoCtx  # same class, but we'll pass wrong payload_type

    def build_agent(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    def build_deps(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


# ── helper workflow to run the channel ──────────────────────────────────────

# Module-level spy dict — channel delivers into here, test body reads it.
_DELIVERIES: dict[str, tuple[str, str]] = {}


@Durable.workflow()
async def _flow_with_thread_channel(
    payload_dict: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    payload = _TodoCtx.model_validate(payload_dict)
    chan: ThreadChannel[_TodoCtx] = ThreadChannel(
        helper_agent=_FakeHelperAgent,
        payload_type=_TodoCtx,
    )
    timeout = (
        timedelta(seconds=timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    verdict: CardVerdict[_TodoCtx] = await chan.request(payload, timeout=timeout)
    return verdict.model_dump(mode="json")


async def _wait(spy: dict, want: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if len(spy) >= want:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("spy never reached expected count")


# ── constructor guard tests (no DBOS needed) ─────────────────────────────────

def test_constructor_raises_when_metadata_model_is_none() -> None:
    with pytest.raises(ValueError, match="metadata_model is None"):
        ThreadChannel(
            helper_agent=_AgentWithoutMetadataModel,
            payload_type=_TodoCtx,
        )


def test_constructor_raises_when_metadata_model_mismatches_payload_type() -> None:
    class _OtherCtx(BaseModel):
        x: str

    with pytest.raises(TypeError, match="must equal"):
        ThreadChannel(
            helper_agent=_AgentWithWrongMetadataModel,
            payload_type=_OtherCtx,
        )


# ── integration tests (need DBOS runtime) ────────────────────────────────────

@pytest.mark.asyncio
async def test_thread_channel_happy_path(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deliver() spies on helper_thread_created; then send_async returns verdict."""
    _DELIVERIES.clear()
    signal_received: list[dict[str, Any]] = []

    # Spy on _create_helper_thread — return a stable fake UUID, record args.
    fake_thread_id = uuid4()

    async def _fake_create_helper_thread(
        *, agent_name: str, metadata: dict[str, Any]
    ) -> UUID:
        _DELIVERIES[metadata.get("request_id", "")] = (
            metadata.get("workflow_id", ""),
            metadata.get("respond_topic", ""),
        )
        return fake_thread_id

    monkeypatch.setattr(
        "ballast.patterns.hitl.channels._thread_plumbing._create_helper_thread",
        _fake_create_helper_thread,
    )

    # Spy on helper_thread_created signal
    async def _on_signal(_sender: Any, **kwargs: Any) -> None:
        signal_received.append(kwargs)

    helper_thread_created.connect(_on_signal)
    try:
        payload = _TodoCtx(
            proposed_title="Buy milk",
            proposed_body="2 litres",
            parent_thread_id="parent-123",
        )
        handle = await Durable.start_workflow(
            _flow_with_thread_channel,
            payload.model_dump(mode="json"),
        )

        # Wait until deliver has been called
        await _wait(_DELIVERIES, want=1)
        (wf_id, respond_topic), = _DELIVERIES.values()

        # Simulate the helper agent tool calling send_async to respond
        verdict = CardVerdict[_TodoCtx](
            decision="approve",
            modified=None,
            feedback=None,
            answered_at=datetime.now(UTC),
        )
        await Durable.send_async(
            destination_id=wf_id,
            topic=respond_topic,
            message=verdict.model_dump(mode="json"),
        )

        result = await handle.get_result()
        assert result["decision"] == "approve"
    finally:
        helper_thread_created.disconnect(_on_signal)


@pytest.mark.asyncio
async def test_thread_channel_timeout(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No send_async within tight deadline → TimeoutError from decode_verdict."""
    fake_thread_id = uuid4()
    deliveries: dict[str, Any] = {}

    async def _fake_create_helper_thread(
        *, agent_name: str, metadata: dict[str, Any]
    ) -> UUID:
        deliveries[metadata.get("request_id", "")] = metadata
        return fake_thread_id

    monkeypatch.setattr(
        "ballast.patterns.hitl.channels._thread_plumbing._create_helper_thread",
        _fake_create_helper_thread,
    )

    payload = _TodoCtx(
        proposed_title="x",
        proposed_body="y",
        parent_thread_id="p",
    )
    handle = await Durable.start_workflow(
        _flow_with_thread_channel,
        payload.model_dump(mode="json"),
        timeout_seconds=0.5,
    )

    with pytest.raises(Exception):
        await handle.get_result()


@pytest.mark.asyncio
async def test_thread_channel_deliver_notifies_parent_when_scoped(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When progress_to_thread is active, helper_thread_created fires with
    the parent_thread_id populated."""
    from ballast.events.context import progress_to_thread

    fake_thread_id = uuid4()
    deliveries: dict[str, Any] = {}

    async def _fake_create_helper_thread(
        *, agent_name: str, metadata: dict[str, Any]
    ) -> UUID:
        deliveries["meta"] = metadata
        return fake_thread_id

    monkeypatch.setattr(
        "ballast.patterns.hitl.channels._thread_plumbing._create_helper_thread",
        _fake_create_helper_thread,
    )

    signal_events: list[dict[str, Any]] = []

    async def _capture(sender: Any, **kwargs: Any) -> None:
        signal_events.append(kwargs)

    helper_thread_created.connect(_capture)
    try:
        chan: ThreadChannel[_TodoCtx] = ThreadChannel(
            helper_agent=_FakeHelperAgent,
            payload_type=_TodoCtx,
        )
        parent_tid = str(uuid4())
        with progress_to_thread(UUID(parent_tid)):
            await chan.deliver(
                request_id="req-42",
                workflow_id="wf-42",
                respond_topic="hitl:req-42",
                payload=_TodoCtx(
                    proposed_title="t",
                    proposed_body="b",
                    parent_thread_id=parent_tid,
                ),
            )

        assert len(signal_events) == 1
        ev = signal_events[0]
        assert ev["helper_agent_name"] == _FakeHelperAgent.name
        assert ev["helper_thread_id"] == fake_thread_id
    finally:
        helper_thread_created.disconnect(_capture)

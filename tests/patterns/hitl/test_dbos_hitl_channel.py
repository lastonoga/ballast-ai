"""``DBOSHITLChannel`` ABC — request() = deliver + recv + decode_verdict."""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.durable import Durable
from ballast.patterns.hitl.channels import DBOSHITLChannel


class _Payload(BaseModel):
    title: str


class _Verdict(BaseModel):
    decision: str


# Spy slot — child writes (workflow_id, topic) so the test body can
# dial send_async at the right destination.
_DELIVERIES: dict[str, tuple[str, str]] = {}


class _SpyChannel(DBOSHITLChannel[_Payload, _Verdict]):
    async def deliver(self, *, request_id, workflow_id,
                      respond_topic, payload) -> None:
        _DELIVERIES[request_id] = (workflow_id, respond_topic)

    async def decode_verdict(self, raw: Any) -> _Verdict:
        return _Verdict.model_validate(raw)


@Durable.workflow()
async def _flow(payload: _Payload, request_id: str) -> _Verdict:
    chan = _SpyChannel()
    return await chan.request(payload, timeout=None)


async def _wait(rid_seen_in: dict, want: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if len(rid_seen_in) >= want:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError


@pytest.mark.asyncio
async def test_request_delivers_then_decodes_recv(
    fresh_dbos_executor: None,
) -> None:
    _DELIVERIES.clear()
    handle = await Durable.start_workflow(
        _flow, _Payload(title="x"), str(uuid4()),
    )
    await _wait(_DELIVERIES, want=1)
    (wfid, topic), = _DELIVERIES.values()

    await Durable.send_async(
        destination_id=wfid, topic=topic,
        message={"decision": "approve"},
    )
    result = await handle.get_result()
    assert isinstance(result, _Verdict)
    assert result.decision == "approve"


@pytest.mark.asyncio
async def test_abstract_methods_must_be_overridden() -> None:
    with pytest.raises(TypeError, match="abstract"):
        DBOSHITLChannel()  # type: ignore[abstract]

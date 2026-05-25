"""Spike: validate the HITLChannel wire end-to-end.

The proposed `channel.request()` shape requires:

  1. A child `@Durable.workflow` that calls `Durable.recv_async(topic)`
     can be invoked from a parent workflow.
  2. From OUTSIDE any workflow (e.g. a REST handler), we can call
     `Durable.send_async(destination_id=<child_id>, topic=...)` and
     the child unblocks with the verdict.
  3. The parent workflow correctly receives the child's return value.
  4. Two invocation styles work:
       a. Direct call: `await child_flow(payload)` from inside parent.
       b. Explicit:    `Durable.start_workflow(child_flow, payload)`.

If either style fails, the spec must adapt before we write the plan.

Spike rules: no UI, no router, no repo — just the recv/send wire.
The "channel" is a stub that records (workflow_id, topic) into a
module-level dict so the test body can address `send_async` at the
right child.
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from dbos import SetWorkflowID
from pydantic import BaseModel

from ballast.durable import Durable

# Spy slot: child workflows write (workflow_id, topic) here.
_DELIVERIES: dict[str, tuple[str, str]] = {}


class ProposedNote(BaseModel):
    title: str
    body: str


class CardVerdict(BaseModel):
    decision: str            # "approve" | "reject"
    modified: ProposedNote | None = None


# ── Style A: direct call `await child_flow(payload)` ─────────────────


@Durable.step()
async def _spy_deliver_a(request_id: str, topic: str) -> None:
    """Step so deliver memoises across replay (mirrors DBOSHITLChannel)."""
    workflow_id = Durable.current_workflow_id()
    _DELIVERIES[request_id] = (workflow_id, topic)


@Durable.workflow()
async def child_flow_direct(
    payload: ProposedNote, request_id: str,
) -> CardVerdict:
    topic = f"hitl:{request_id}"
    await _spy_deliver_a(request_id, topic)
    raw = await Durable.recv_async(topic, timeout_seconds=10.0)
    assert raw is not None, "child timed out waiting for verdict"
    return CardVerdict.model_validate(raw)


@Durable.workflow()
async def parent_flow_direct(
    payload: ProposedNote, request_id: str,
) -> dict[str, Any]:
    verdict = await child_flow_direct(payload, request_id)
    # Parent makes a decision based on the typed verdict.
    if verdict.decision != "approve":
        return {"saved": False, "reason": "rejected"}
    final = verdict.modified or payload
    return {
        "saved": True, "title": final.title, "body": final.body,
    }


# ── Style B: explicit `Durable.start_workflow(child_flow, ...)` ──────


@Durable.step()
async def _spy_deliver_b(request_id: str, topic: str) -> None:
    workflow_id = Durable.current_workflow_id()
    _DELIVERIES[request_id] = (workflow_id, topic)


@Durable.workflow()
async def child_flow_explicit(
    payload: ProposedNote, request_id: str,
) -> CardVerdict:
    topic = f"hitl:{request_id}"
    await _spy_deliver_b(request_id, topic)
    raw = await Durable.recv_async(topic, timeout_seconds=10.0)
    assert raw is not None
    return CardVerdict.model_validate(raw)


@Durable.workflow()
async def parent_flow_explicit(
    payload: ProposedNote, request_id: str, child_wfid: str,
) -> dict[str, Any]:
    with SetWorkflowID(child_wfid):
        handle = await Durable.start_workflow(
            child_flow_explicit, payload, request_id,
        )
    verdict = await handle.get_result()
    if verdict.decision != "approve":
        return {"saved": False, "reason": "rejected"}
    final = verdict.modified or payload
    return {
        "saved": True, "title": final.title, "body": final.body,
    }


# ── Helpers ──────────────────────────────────────────────────────────


async def _wait_for_delivery(request_id: str, timeout: float = 5.0) -> tuple[str, str]:
    """Poll until the child writes (workflow_id, topic) into the spy."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if request_id in _DELIVERIES:
            return _DELIVERIES[request_id]
        await asyncio.sleep(0.05)
    raise TimeoutError(f"no delivery for {request_id}")


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_style_A_direct_call_approve(
    fresh_dbos_executor: None,
) -> None:
    """Parent calls `await child_flow(payload)` directly.

    Validates that:
      - Child has its OWN workflow_id (not parent's).
      - send_async from outside reaches that child.
      - Parent's `await` resumes with the typed CardVerdict.
    """
    _DELIVERIES.clear()
    request_id = str(uuid4())
    payload = ProposedNote(title="grocery", body="milk")

    # Kick off parent in background so we can dial send_async from the
    # test (the outside) before the child times out on recv.
    parent_handle = await Durable.start_workflow(
        parent_flow_direct, payload, request_id,
    )
    child_wfid, topic = await _wait_for_delivery(request_id)

    # Critical assertion: child has its own id, distinct from parent.
    parent_wfid = parent_handle.workflow_id
    assert child_wfid != parent_wfid, (
        f"child_flow_direct ran INLINED into parent (both ids = "
        f"{parent_wfid}). Spec assumption broken — direct `await "
        f"child_flow(...)` does NOT spawn a child workflow."
    )

    # Send the verdict from outside any workflow.
    verdict = CardVerdict(
        decision="approve",
        modified=ProposedNote(title="Grocery — Sat", body="milk, eggs"),
    )
    await Durable.send_async(
        destination_id=child_wfid,
        message=verdict.model_dump(mode="json"),
        topic=topic,
    )

    result = await parent_handle.get_result()
    assert result == {
        "saved": True, "title": "Grocery — Sat", "body": "milk, eggs",
    }


@pytest.mark.asyncio
async def test_style_A_direct_call_reject(
    fresh_dbos_executor: None,
) -> None:
    """Same wire, decision='reject' → parent returns the reject branch."""
    _DELIVERIES.clear()
    request_id = str(uuid4())
    parent_handle = await Durable.start_workflow(
        parent_flow_direct, ProposedNote(title="x", body="y"), request_id,
    )
    child_wfid, topic = await _wait_for_delivery(request_id)
    await Durable.send_async(
        destination_id=child_wfid,
        message=CardVerdict(decision="reject").model_dump(mode="json"),
        topic=topic,
    )
    result = await parent_handle.get_result()
    assert result == {"saved": False, "reason": "rejected"}


@pytest.mark.asyncio
async def test_style_B_start_workflow_approve(
    fresh_dbos_executor: None,
) -> None:
    """Parent spawns child via `Durable.start_workflow` with explicit
    child workflow_id. This is the fallback style if direct call
    doesn't give us a distinct child id.
    """
    _DELIVERIES.clear()
    request_id = str(uuid4())
    child_wfid = f"child-{uuid4()}"
    parent_handle = await Durable.start_workflow(
        parent_flow_explicit,
        ProposedNote(title="grocery", body="milk"),
        request_id,
        child_wfid,
    )
    delivered_wfid, topic = await _wait_for_delivery(request_id)
    assert delivered_wfid == child_wfid, (
        "child reported a different workflow_id than the one we set"
    )

    await Durable.send_async(
        destination_id=child_wfid,
        message=CardVerdict(decision="approve").model_dump(mode="json"),
        topic=topic,
    )
    result = await parent_handle.get_result()
    assert result == {"saved": True, "title": "grocery", "body": "milk"}

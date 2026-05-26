# ApprovalCapability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship `ApprovalCapability` that bridges pydantic-ai's `requires_approval=True` tool flow with Ballast's `UICardChannel` HITL infrastructure. Apps wire one capability + a `tool_card_map`, and tools marked `requires_approval=True` automatically trigger HITL approval cards in the UI, with verdicts mapped back to `ToolApproved`/`ToolDenied`.

**Architecture:** Single `ApprovalCapability(BallastCapability)` class with a `wrap_run` hook. The hook calls the underlying `handler()` (normal agent run), inspects `result.output` for `DeferredToolRequests`, opens `UICardChannel.request` for each pending approval, awaits verdicts inside the DBOS workflow context (durable), builds `DeferredToolResults`, and re-runs the agent with the resumed state. Loops to handle cascading approval rounds.

**Tech Stack:** Python 3.11+, pydantic v2 (HITL payload models), pydantic-ai (`AbstractCapability.wrap_run`, `DeferredToolRequests`, `DeferredToolResults`, `ToolApproved`, `ToolDenied`, `ToolCallPart`), existing `BallastCapability` / `UICardChannel` / `CardVerdict`.

**Insertion point:** `BallastCapability.wrap_run` — chosen because it executes inside the DBOS workflow context (so `UICardChannel.request` → `Durable.recv_async` works), wraps the full `agent.run()` boundary (so deferred requests are visible), and requires zero changes to either pydantic-ai internals or our existing `DurableAgent`.

---

## Task 1: `ApprovalCapability` core class + tests

**Files:**
- Create: `src/ballast/capabilities/approval.py`
- Create: `tests/capabilities/test_approval.py`

### Step 1: Failing tests

```python
"""ApprovalCapability — bridges pydantic-ai requires_approval to UICardChannel."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.result import DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import ToolApproved, ToolDenied

from ballast.capabilities.approval import ApprovalCapability
from ballast.patterns.hitl import CardVerdict


# --- Test payloads ---------------------------------------------------------

class _PublishPayload(BaseModel):
    __hitl_kind__: ClassVar[str] = "test-publish"
    title: str
    body: str


class _PublishVerdict(CardVerdict[_PublishPayload]):
    __hitl_kind__: ClassVar[str] = "test-publish"


# --- Fakes ------------------------------------------------------------------

@dataclass
class _FakeAgentResult:
    output: Any
    _messages: list = None
    def all_messages(self) -> list: return self._messages or []


class _FakeChannel:
    """Stand-in for UICardChannel — records requests, returns scripted verdicts."""
    def __init__(self, verdicts: list[CardVerdict]):
        self.verdicts = list(verdicts)
        self.requests: list = []

    async def request(self, payload, *, timeout=None):
        self.requests.append({"payload": payload, "timeout": timeout})
        return self.verdicts.pop(0)


# --- Tests ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wrap_run_passes_through_when_no_deferred_requests() -> None:
    """If agent.run returns a normal output, capability is transparent."""
    cap = ApprovalCapability(tool_card_map={})
    final_result = _FakeAgentResult(output="normal text response")

    handler = AsyncMock(return_value=final_result)
    handler.__self__ = AsyncMock()  # would be the Agent in real flow

    fake_ctx = AsyncMock()
    out = await cap.wrap_run(fake_ctx, handler=handler)
    assert out is final_result
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_wrap_run_opens_card_for_each_deferred_approval() -> None:
    """For each ToolCallPart in DeferredToolRequests.approvals, channel.request fires."""
    captured_payloads = []

    def make_payload(tc: ToolCallPart, deps: Any) -> _PublishPayload:
        captured_payloads.append(tc)
        args = tc.args if isinstance(tc.args, dict) else {}
        return _PublishPayload(title=args.get("title", ""), body=args.get("body", ""))

    channel = _FakeChannel([
        _PublishVerdict(decision="approve", modified=None, feedback=None, answered_at=None),
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

    # Second call (agent.run with deferred_tool_results) received correct mapping
    fake_agent.run.assert_awaited_once()
    call_kwargs = fake_agent.run.await_args.kwargs
    deferred_results: DeferredToolResults = call_kwargs["deferred_tool_results"]
    approvals = deferred_results.approvals
    assert "tc-1" in approvals
    assert isinstance(approvals["tc-1"], ToolApproved)


@pytest.mark.asyncio
async def test_wrap_run_maps_reject_to_tool_denied() -> None:
    """CardVerdict.decision='reject' → ToolDenied with feedback message."""
    channel = _FakeChannel([
        _PublishVerdict(
            decision="reject", modified=None,
            feedback="not now", answered_at=None,
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
    """CardVerdict.modified payload → ToolApproved(override_args=<dict>)."""
    modified_payload = _PublishPayload(title="EDITED", body="EDITED-BODY")
    channel = _FakeChannel([
        _PublishVerdict(
            decision="approve", modified=modified_payload,
            feedback=None, answered_at=None,
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
    """Tool name not in tool_card_map → auto-deny with explanation."""
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
```

### Step 2: Run — confirm fail (ImportError).

### Step 3: Implement `src/ballast/capabilities/approval.py`

```python
"""``ApprovalCapability`` — bridges pydantic-ai requires_approval flow with UICardChannel.

When an agent's tools are marked ``requires_approval=True`` in pydantic-ai,
the agent.run() completes with ``result.output: DeferredToolRequests`` —
the tools didn't execute; instead, the app must collect human verdicts
and re-call agent.run() with ``deferred_tool_results=...``.

This capability automates the loop: for each approval in the deferred
requests, it opens a HITL card via the configured channel, awaits the
verdict (inside the DBOS workflow context, so it's durable), then maps
the verdict to ``ToolApproved`` or ``ToolDenied`` and re-runs the agent.
Cascading approval rounds are handled by an outer ``while`` loop.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.result import AgentRunResult, DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import ToolApproved, ToolDenied

from ballast.capabilities.base import BallastCapability

CardFactory = Callable[[ToolCallPart, Any], BaseModel]
"""Builds the HITL card payload from a tool-call part + agent deps."""

ToolEntry = tuple[type[BaseModel], CardFactory, Any]
"""(payload_model_class, factory, channel) — channel is HITLChannel-compatible."""


class ApprovalCapability(BallastCapability):
    """Bridges requires_approval=True tools to UICardChannel HITL flow.

    Apps wire one capability per agent + a ``tool_card_map`` keyed by
    tool name. Tools NOT in the map auto-deny with a helpful message.
    """

    name = "approval_capability"

    def __init__(
        self, *,
        tool_card_map: Mapping[str, ToolEntry],
        timeout: timedelta | None = None,
    ) -> None:
        self._tool_card_map = dict(tool_card_map)
        self._timeout = timeout

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: Callable[[], Any],
    ) -> AgentRunResult[Any]:
        """Run the agent; loop on DeferredToolRequests until final output."""
        result = await handler()

        while isinstance(result.output, DeferredToolRequests):
            approvals: dict[str, ToolApproved | ToolDenied] = {}

            for tc in result.output.approvals:
                approvals[tc.tool_call_id] = await self._resolve_approval(tc, ctx.deps)

            # The handler is bound to a method on the underlying Agent; reach
            # the Agent via handler.__self__ to call .run() with resumed state.
            agent = handler.__self__  # type: ignore[attr-defined]
            deferred_results = DeferredToolResults(
                approvals=approvals, calls={},
            )
            result = await agent.run(
                None,
                message_history=result.all_messages(),
                deferred_tool_results=deferred_results,
                deps=ctx.deps,
            )

        return result

    async def _resolve_approval(
        self, tc: ToolCallPart, deps: Any,
    ) -> ToolApproved | ToolDenied:
        entry = self._tool_card_map.get(tc.tool_name)
        if entry is None:
            return ToolDenied(
                message=(
                    f"Tool {tc.tool_name!r} is not registered in tool_card_map; "
                    "auto-denying. Either add it to ApprovalCapability.tool_card_map "
                    "or remove requires_approval=True from the tool."
                ),
            )

        payload_cls, factory, channel = entry
        payload = factory(tc, deps)
        verdict = await channel.request(payload, timeout=self._timeout)

        if verdict.decision == "approve":
            override_args = None
            if verdict.modified is not None:
                override_args = verdict.modified.model_dump()
            return ToolApproved(override_args=override_args)

        return ToolDenied(message=verdict.feedback or "Denied by user.")


__all__ = ["ApprovalCapability", "CardFactory", "ToolEntry"]
```

### Step 4: Run — confirm pass

Run: `uv run pytest tests/capabilities/test_approval.py -v`
Expected: 5 passed.

### Step 5: Commit

```bash
git add src/ballast/capabilities/approval.py tests/capabilities/test_approval.py
git commit -m "feat(capabilities): ApprovalCapability — bridge pydantic-ai requires_approval to UICardChannel"
```

## Notes
- `DeferredToolResults(approvals=..., calls=...)` — both required. Pass `calls={}` if no external tool calls.
- `agent.run()` resume signature varies across pydantic-ai versions. If the kwargs don't match (`message_history` may be positional, `deferred_tool_results` name, etc.), inspect `Agent.run` source in `.venv/lib/python*/site-packages/pydantic_ai/agent.py` and adjust.
- `handler.__self__` accesses the bound Agent. In pydantic-ai's capability flow this attribute exists; if not, walk `handler.__closure__` or accept the agent explicitly via constructor.
- Verdict shape (`decision: "approve"|"reject"`, `modified: BaseModel | None`, `feedback: str | None`) — confirm in `src/ballast/patterns/hitl/card_verdict.py` (or wherever `CardVerdict` lives). Adjust if field names differ.

---

## Task 2: Public API re-exports

**Files:**
- Modify: `src/ballast/capabilities/__init__.py`
- Modify: `src/ballast/__init__.py`

### Step 1: Update capability subpackage __init__

Read `src/ballast/capabilities/__init__.py`. Add `ApprovalCapability` import + entry in `__all__` (alphabetical).

### Step 2: Top-level __init__

Read `src/ballast/__init__.py`. Find existing top-level capability re-exports (e.g. `BudgetGuard`, `GoalDriftDetector`). Add `ApprovalCapability` import + entry in `__all__` (alphabetical).

### Step 3: Smoke import

```
uv run python -c "from ballast import ApprovalCapability; print('ok')"
```
Expected: `ok`.

### Step 4: Run full suite

Run: `uv run pytest tests/ -q`
Expected: green.

### Step 5: Commit

```bash
git add src/ballast/__init__.py src/ballast/capabilities/__init__.py
git commit -m "feat(ballast): re-export ApprovalCapability at top level"
```

---

## Task 3: Final smoke

- [ ] **Step 1: Run framework + capability suites**

```
uv run pytest tests/ -q
uv run pytest tests/capabilities/test_approval.py -v
```
Expected: green; 5 ApprovalCapability tests pass.

- [ ] **Step 2: Smoke imports**

```
uv run python -c "
from ballast import (
    ApprovalCapability,
    CircuitBreaker, PlanAndExecute, Scored,
    GoalDriftDetector,
)
print('all imports ok')
"
```

- [ ] **Step 3: Optional cleanup commit**

```bash
git status
git add -u && git commit -m "chore(approval-capability): final smoke" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- Task 1 ships the core class + 5 unit tests covering happy path / reject / modified override / unmapped denial / pass-through.
- Task 2 wires public API.
- Task 3 final smoke.

**Placeholder scan:** No TBDs. Every step has complete code or exact command. Task 1 notes potential pydantic-ai version-specific adjustments to `agent.run()` resume kwargs — implementer must verify against installed pydantic-ai.

**Type consistency:**
- `tool_card_map: dict[str, ToolEntry]` where `ToolEntry = (payload_cls, factory, channel)` — consistent across tests + impl.
- `CardVerdict[T]` field names (`decision`, `modified`, `feedback`) — verified against `ballast.patterns.hitl.CardVerdict` (the tests reference these directly).
- `ToolApproved(override_args=...)` / `ToolDenied(message=...)` — pydantic-ai standard API.

**Known plan-vs-implementation deviation point:** `agent.run(deferred_tool_results=..., message_history=..., deps=...)` kwargs may differ slightly. Implementer flagged to inspect actual signature and adapt. Tests use `AsyncMock` so they don't care about real signature; impl must match runtime API.

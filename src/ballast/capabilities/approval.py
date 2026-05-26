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
from pydantic_ai import AgentRunResult
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

from ballast.capabilities.base import BallastCapability

CardFactory = Callable[[ToolCallPart, Any], BaseModel]
ToolEntry = tuple[type[BaseModel], CardFactory, Any]


class ApprovalCapability(BallastCapability):
    """Bridges requires_approval=True tools to UICardChannel HITL flow."""

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

            agent = handler.__self__  # type: ignore[attr-defined]
            deferred_results = DeferredToolResults(approvals=approvals, calls={})
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

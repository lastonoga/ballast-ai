from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic_ai import Agent, RunContext

from pydantic_ai_stateflow.patterns.hitl.response import (
    ApprovedResponse,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
)
from pydantic_ai_stateflow.patterns.hitl.verdict import HelperVerdict


@dataclass
class HelperToolBox:
    """Mutable container the approval tools write into.

    The session runner (Task 7) inspects `response` after each `agent.run`;
    if non-None, the conversation is complete and the response is sent to
    the gate's topic.
    """

    response: HITLResponse | None = None


@dataclass
class HelperDeps:
    """`ctx.deps` shape for the helper agent (`Agent[HelperDeps, str]`)."""

    request_id: UUID
    tenant_id: UUID
    actor_id: str
    turn_count: int
    tools_invoked_so_far: list[str]
    toolbox: HelperToolBox
    autopilot_eligible: bool = False
    cached_recommendation_confidence: float | None = None


@runtime_checkable
class HelperAgentFactory(Protocol):
    """Builds a HelperAgent from a base pydantic-ai Agent + a request_id.

    Default impl is `make_helper_agent_with_approval_tools`. Apps may
    supply alternatives that register custom tools.
    """

    def __call__(
        self,
        *,
        base_agent: Agent[HelperDeps, str],
        request_id: UUID,
        context_type: type[Any] | None = None,
        allow_modify: bool = False,
        allow_partial: bool = False,
    ) -> Agent[HelperDeps, str]: ...


def make_helper_agent_with_approval_tools(
    *,
    base_agent: Agent[HelperDeps, str],
    request_id: UUID,
    context_type: type[Any] | None = None,
    allow_modify: bool = False,
    allow_partial: bool = False,
) -> Agent[HelperDeps, str]:
    """Register approve / reject (+ optional modify / finalize_partial) tools.

    Each tool builds the appropriate `HITLResponse` + `HelperVerdict[context_type]`
    and writes it to `ctx.deps.toolbox.response`. The session runner (Task 7)
    picks it up there. Mutating the toolbox is preferred over `DBOS.send` here
    because the runner is the boundary that decides whether to send (single
    source of truth — keeps the factory testable without DBOS).
    """

    verdict_type: Any = (
        HelperVerdict[context_type] if context_type is not None  # type: ignore[valid-type]
        else HelperVerdict[None]
    )

    def _build_verdict(
        ctx: RunContext[HelperDeps], *, rationale: str, confidence: float,
        context: Any | None = None,
    ) -> dict[str, Any]:
        verdict = verdict_type(
            rationale=rationale,
            confidence=confidence,
            conversation_turn_count=ctx.deps.turn_count,
            tools_invoked=list(ctx.deps.tools_invoked_so_far),
            autopilot_eligible=ctx.deps.autopilot_eligible,
            autopilot_confidence=ctx.deps.cached_recommendation_confidence,
            context=context,
        )
        dumped: dict[str, Any] = verdict.model_dump(mode="json")
        return dumped

    if context_type is not None:
        async def approve_with_context(
            ctx: RunContext[HelperDeps],
            rationale: str,
            confidence: float,
            context: Any,
        ) -> str:
            ctx.deps.toolbox.response = ApprovedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=confidence,
                    context=context,
                ),
            )
            return "approved"

        approve_with_context.__name__ = "approve"
        approve_with_context.__annotations__ = {
            "ctx": RunContext[HelperDeps],
            "rationale": str,
            "confidence": float,
            "context": context_type,
            "return": str,
        }
        base_agent.tool(approve_with_context)
    else:
        async def approve_no_context(
            ctx: RunContext[HelperDeps],
            rationale: str,
            confidence: float,
        ) -> str:
            ctx.deps.toolbox.response = ApprovedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=confidence,
                ),
            )
            return "approved"

        approve_no_context.__name__ = "approve"
        approve_no_context.__annotations__ = {
            "ctx": RunContext[HelperDeps],
            "rationale": str,
            "confidence": float,
            "return": str,
        }
        base_agent.tool(approve_no_context)

    @base_agent.tool
    async def reject(
        ctx: RunContext[HelperDeps],
        rationale: str,
        feedback: str | None = None,
    ) -> str:
        ctx.deps.toolbox.response = RejectedResponse(
            actor_id=ctx.deps.actor_id,
            answered_at=datetime.now(tz=UTC),
            feedback=feedback or rationale,
            helper_verdict=_build_verdict(
                ctx, rationale=rationale, confidence=1.0,
            ),
        )
        return "rejected"

    if allow_modify:
        @base_agent.tool
        async def modify(
            ctx: RunContext[HelperDeps],
            rationale: str,
            confidence: float,
            modified_proposal: dict[str, Any],
        ) -> str:
            ctx.deps.toolbox.response = ModifiedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                modified_proposal=modified_proposal,
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=confidence,
                ),
            )
            return "modified"

    if allow_partial:
        @base_agent.tool
        async def finalize_partial(
            ctx: RunContext[HelperDeps],
            rationale: str,
            approved_element_ids: list[str],
            rejected_element_ids: list[str],
            modifications: dict[str, dict[str, Any]] | None = None,
        ) -> str:
            ctx.deps.toolbox.response = ModifiedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                modified_proposal={
                    "__partial__": True,
                    "approved_element_ids": list(approved_element_ids),
                    "rejected_element_ids": list(rejected_element_ids),
                    "modifications": dict(modifications or {}),
                },
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=1.0,
                ),
            )
            return "partial"

    return base_agent

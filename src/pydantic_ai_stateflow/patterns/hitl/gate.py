from __future__ import annotations

import itertools
from typing import Any, ClassVar
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance

from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.patterns.errors import HITLDenied, HITLTimedOut
from pydantic_ai_stateflow.patterns.hitl.channel import HITLChannel
from pydantic_ai_stateflow.patterns.hitl.policy import Policy
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.persistence import HITLRepository

_KIND_TO_VERDICT = {
    "approved": "approve",
    "rejected": "reject",
    "modified": "revise",
}

_instance_counter = itertools.count()


@DBOS.dbos_class()
class HITLGate(DBOSConfiguredInstance):
    """HITL pause + authz (spec 2C.4 + Critical Fix #2).

    Authz happens at TWO points:

    1. ENDPOINT-side (FastAPI / Slack handler): `policy.can(...)` is checked
       BEFORE the responder reaches the workflow's recv topic. Unauthorized
       responses never appear here.

    2. WORKFLOW-side (this Pattern, defense-in-depth): on receive, we re-run
       `policy.can(...)`. Endpoints may be bypassed by future channels;
       this check ensures *every* path through the gate is verified.

    Denied attempts are persisted to `hitl_authz_denials` (via repo) so
    audit trails are complete.

    `name` and `run` satisfy `Pattern[HITLPrompt, HITLResponse]`. The
    `ask` shortcut adapts to the `Asker` Protocol (spec 4A.0.4) so
    policies like EscalateToHITLOnReject can depend on Asker instead of
    importing HITLGate concretely.
    """

    name: ClassVar[str] = "hitl_gate"

    def __init__(
        self,
        *,
        channel: HITLChannel,
        policy: Policy,
        repo: HITLRepository,
    ) -> None:
        super().__init__(config_name=f"hitl-gate-{next(_instance_counter)}")
        self.channel = channel
        self.policy = policy
        self.repo = repo

    @DBOS.workflow()
    @traced(TraceName.PATTERN_HITL_GATE, attrs=lambda self, prompt, *, tenant_id: {
        "tenant_id": str(tenant_id), "pattern": self.name,
    })
    async def run(self, prompt: HITLPrompt, *, tenant_id: UUID) -> HITLResponse:
        if prompt.tenant_id != tenant_id:
            raise ValueError(
                f"HITLGate.run: prompt.tenant_id ({prompt.tenant_id}) does not "
                f"match tenant_id kwarg ({tenant_id})"
            )

        request = await self.repo.persist_request(
            prompt=prompt.model_dump(mode="json"),
            workflow_id=tenant_id,
            gate_kind=self.name,
            purpose="approval",
            tenant_id=tenant_id,
            timeout_at=None,
        )

        response = await self.channel.ask(prompt, request_id=request.id)

        if isinstance(response, TimeoutResponse):
            await self.repo.persist_timeout(request.id, tenant_id=tenant_id)
            raise HITLTimedOut(request_id=request.id)

        verdict = await self.policy.can(
            actor=response.actor_id,
            action="decide",
            resource=prompt,
            tenant_id=tenant_id,
        )
        if not verdict.is_grant:
            await self.repo.persist_authz_denied(
                request_id=request.id,
                actor_id=response.actor_id or "<anonymous>",
                voter_votes=dict(verdict.votes),
                tenant_id=tenant_id,
            )
            raise HITLDenied(
                actor_id=response.actor_id or "<anonymous>",
                votes=dict(verdict.votes),
            )

        helper_verdict_payload: dict[str, Any] | None = None
        helper_verdict_context_type: str | None = None
        helper_thread_id: UUID | None = None
        raw_verdict = getattr(response, "helper_verdict", None)
        if raw_verdict is not None:
            # Mutable copy — `response` is frozen, but the dict itself isn't;
            # copy for clarity and to avoid mutating shared state.
            helper_verdict_payload = dict(raw_verdict)
            # Optional sidecar keys carried inside the helper_verdict blob
            # (set by ConversationalChannel via the helper agent). Stripped
            # from the persisted blob so the row's typed columns hold them.
            tid_str = helper_verdict_payload.pop("__helper_thread_id__", None)
            fqn = helper_verdict_payload.pop("__context_type_fqn__", None)
            if tid_str is not None:
                helper_thread_id = UUID(tid_str)
            if fqn is not None:
                helper_verdict_context_type = fqn

        await self.repo.persist_response(
            request_id=request.id,
            actor_id=response.actor_id or "<anonymous>",
            verdict=_KIND_TO_VERDICT[response.kind],
            payload=response.model_dump(mode="json"),
            tenant_id=tenant_id,
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
        )
        return response

    async def ask(
        self, prompt: HITLPrompt, *, purpose: str = "approval",
    ) -> HITLResponse:
        """Asker Protocol adapter (spec 4A.0.4)."""
        return await self.run(prompt, tenant_id=prompt.tenant_id)

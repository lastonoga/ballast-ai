from __future__ import annotations

import itertools
import uuid as _uuid
from typing import Any, ClassVar, cast
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
    """HITL pause + authz (defense-in-depth on receive).

    Authz is checked twice: at the responder endpoint (so unauthorized
    replies never reach the workflow's recv topic) and again here on
    receive. Denied attempts are persisted to ``hitl_authz_denials``.

    The DBOS workflow id of THIS gate run is stored as
    ``BlockingRequirement.workflow_id`` so the endpoint can route the
    response back via ``DBOS.send(destination_id=workflow_id, ...)``.

    Apps that need tenant/workspace scoping carry it inside their
    ``HITLPrompt`` subclass and the ``Policy`` (which receives
    ``resource=prompt``).
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
    @traced(TraceName.PATTERN_HITL_GATE, attrs=lambda self, prompt: {
        "pattern": self.name,
    })
    async def run(self, prompt: HITLPrompt) -> HITLResponse:
        # Inside a @DBOS.workflow, ``DBOS.workflow_id`` is the current
        # workflow's id — what responders need to ``DBOS.send`` back to.
        # DBOS auto-generated nested workflow ids may not be UUIDs (they
        # take the form ``{parent}-{ordinal}``); coerce non-UUID ids to a
        # deterministic UUID so the BlockingRequirement column accepts them.
        raw_id = cast(str, DBOS.workflow_id)
        try:
            workflow_id = UUID(raw_id)
        except (ValueError, AttributeError, TypeError):
            workflow_id = _uuid.uuid5(_uuid.NAMESPACE_URL, raw_id)

        request = await self.repo.persist_request(
            prompt=prompt.model_dump(mode="json"),
            workflow_id=workflow_id,
            gate_kind=self.name,
            purpose="approval",
            timeout_at=None,
        )

        response = await self.channel.ask(prompt, request_id=request.id)

        if isinstance(response, TimeoutResponse):
            await self.repo.persist_timeout(request.id)
            raise HITLTimedOut(request_id=request.id)

        verdict = await self.policy.can(
            actor=response.actor_id,
            action="decide",
            resource=prompt,
        )
        if not verdict.is_grant:
            await self.repo.persist_authz_denied(
                request_id=request.id,
                actor_id=response.actor_id or "<anonymous>",
                voter_votes=dict(verdict.votes),
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
            helper_verdict_payload = dict(raw_verdict)
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
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
        )
        return response

    async def ask(self, prompt: HITLPrompt) -> HITLResponse:
        """Asker Protocol adapter."""
        return await self.run(prompt)

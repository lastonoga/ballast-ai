from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.hitl.domain import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
    DecisionVerdict,
    _coerce_hitl_purpose,
)


@runtime_checkable
class HITLRepository(Protocol):
    async def persist_request(
        self, *, prompt: dict[str, Any], workflow_id: UUID,
        gate_kind: str, purpose: str, tenant_id: UUID,
        timeout_at: datetime | None = None,
    ) -> BlockingRequirement: ...

    async def load_request(self, request_id: UUID, *, tenant_id: UUID) -> BlockingRequirement | None: ...

    async def persist_response(
        self, *, request_id: UUID, actor_id: str, verdict: str,
        payload: dict[str, Any], tenant_id: UUID,
        helper_verdict_payload: dict[str, Any] | None = None,
        helper_verdict_context_type: str | None = None,
        helper_thread_id: UUID | None = None,
    ) -> Decision: ...

    async def persist_timeout(self, request_id: UUID, *, tenant_id: UUID) -> None: ...

    async def persist_authz_denied(
        self, *, request_id: UUID, actor_id: str,
        voter_votes: dict[str, Any], tenant_id: UUID,
    ) -> AuthzDenial: ...

    async def list_pending(self, *, tenant_id: UUID, limit: int = 100) -> list[BlockingRequirement]: ...


class InMemoryHITLRepository:
    def __init__(self) -> None:
        self._requests: dict[UUID, BlockingRequirement] = {}
        self._decisions: dict[UUID, Decision] = {}
        self._denials: list[AuthzDenial] = []

    async def persist_request(
        self, *, prompt: Any, workflow_id: Any, gate_kind: Any, purpose: Any,
        tenant_id: Any, timeout_at: Any = None,
    ) -> BlockingRequirement:
        req = BlockingRequirement(
            id=uuid4(), tenant_id=tenant_id, gate_kind=gate_kind,
            workflow_id=workflow_id, payload=prompt,
            purpose=_coerce_hitl_purpose(purpose),
            status=BlockingRequirementStatus.PENDING,
            timeout_at=timeout_at, created_at=datetime.now(tz=UTC), resolved_at=None,
        )
        self._requests[req.id] = req
        return req

    async def load_request(self, request_id: Any, *, tenant_id: Any) -> BlockingRequirement | None:
        req = self._requests.get(request_id)
        if req is None or req.tenant_id != tenant_id:
            return None
        return req

    async def persist_response(
        self, *, request_id: Any, actor_id: Any, verdict: Any, payload: Any,
        tenant_id: Any, helper_verdict_payload: Any = None,
        helper_verdict_context_type: Any = None, helper_thread_id: Any = None,
    ) -> Decision:
        req = self._requests.get(request_id)
        if req is None or req.tenant_id != tenant_id:
            raise KeyError(f"Request {request_id} not found")
        dec = Decision(
            id=uuid4(), tenant_id=tenant_id,
            blocking_requirement_id=request_id, actor_id=actor_id,
            verdict=DecisionVerdict(verdict), payload=payload,
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
            created_at=datetime.now(tz=UTC),
        )
        self._decisions[dec.id] = dec
        self._requests[request_id] = req.model_copy(update={
            "status": BlockingRequirementStatus.RESOLVED,
            "resolved_at": datetime.now(tz=UTC),
        })
        return dec

    async def persist_timeout(self, request_id: Any, *, tenant_id: Any) -> None:
        req = self._requests.get(request_id)
        if req is None or req.tenant_id != tenant_id:
            return
        self._requests[request_id] = req.model_copy(update={
            "status": BlockingRequirementStatus.TIMED_OUT,
            "resolved_at": datetime.now(tz=UTC),
        })

    async def persist_authz_denied(
        self, *, request_id: Any, actor_id: Any, voter_votes: Any, tenant_id: Any,
    ) -> AuthzDenial:
        denial = AuthzDenial(
            id=uuid4(), tenant_id=tenant_id, request_id=request_id,
            actor_id=actor_id, voter_votes=dict(voter_votes),
            attempted_at=datetime.now(tz=UTC),
        )
        self._denials.append(denial)
        return denial

    async def list_pending(self, *, tenant_id: Any, limit: int = 100) -> list[BlockingRequirement]:
        return [r for r in self._requests.values()
                if r.tenant_id == tenant_id
                and r.status == BlockingRequirementStatus.PENDING][:limit]

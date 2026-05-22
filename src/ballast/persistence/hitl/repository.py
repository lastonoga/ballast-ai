from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from ballast.persistence.hitl.domain import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
    DecisionVerdict,
)


@runtime_checkable
class HITLRepository(Protocol):
    async def persist_request(
        self,
        *,
        prompt: dict[str, Any],
        workflow_id: UUID,
        gate_kind: str,
        purpose: str,
        timeout_at: datetime | None = None,
        request_id: UUID | None = None,
    ) -> BlockingRequirement:
        """Persist a new blocking requirement.

        ``request_id`` lets the caller pre-allocate the id (e.g. so it
        can be embedded in a child-thread's metadata BEFORE the request
        is persisted). When ``None``, the repo generates one.
        """
        ...

    async def load_request(
        self, request_id: UUID,
    ) -> BlockingRequirement | None: ...

    async def persist_response(
        self,
        *,
        request_id: UUID,
        actor_id: str,
        verdict: str,
        payload: dict[str, Any],
        helper_verdict_payload: dict[str, Any] | None = None,
        helper_verdict_context_type: str | None = None,
        helper_thread_id: UUID | None = None,
    ) -> Decision: ...

    async def persist_timeout(self, request_id: UUID) -> None: ...

    async def persist_authz_denied(
        self,
        *,
        request_id: UUID,
        actor_id: str,
        voter_votes: dict[str, Any],
    ) -> AuthzDenial: ...

    async def list_pending(self, *, limit: int = 100) -> list[BlockingRequirement]: ...


class InMemoryHITLRepository:
    def __init__(self) -> None:
        self._requests: dict[UUID, BlockingRequirement] = {}
        self._decisions: dict[UUID, Decision] = {}
        self._denials: list[AuthzDenial] = []

    async def persist_request(
        self, *, prompt: Any, workflow_id: Any, gate_kind: Any,
        purpose: Any, timeout_at: Any = None,
        request_id: UUID | None = None,
    ) -> BlockingRequirement:
        req = BlockingRequirement(
            id=request_id if request_id is not None else uuid4(),
            gate_kind=gate_kind,
            workflow_id=workflow_id,
            payload=dict(prompt),
            purpose=str(purpose),
            status=BlockingRequirementStatus.PENDING,
            timeout_at=timeout_at,
            created_at=datetime.now(tz=UTC),
            resolved_at=None,
        )
        self._requests[req.id] = req
        return req

    async def load_request(
        self, request_id: Any,
    ) -> BlockingRequirement | None:
        return self._requests.get(request_id)

    async def persist_response(
        self, *, request_id: Any, actor_id: Any, verdict: Any, payload: Any,
        helper_verdict_payload: Any = None,
        helper_verdict_context_type: Any = None, helper_thread_id: Any = None,
    ) -> Decision:
        req = self._requests.get(request_id)
        if req is None:
            raise KeyError(f"Request {request_id} not found")
        dec = Decision(
            id=uuid4(),
            blocking_requirement_id=request_id,
            actor_id=actor_id,
            verdict=DecisionVerdict(verdict),
            payload=dict(payload),
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
            created_at=datetime.now(tz=UTC),
        )
        self._decisions[dec.id] = dec
        req.status = BlockingRequirementStatus.RESOLVED
        req.resolved_at = datetime.now(tz=UTC)
        return dec

    async def persist_timeout(self, request_id: Any) -> None:
        req = self._requests.get(request_id)
        if req is None:
            return
        req.status = BlockingRequirementStatus.TIMED_OUT
        req.resolved_at = datetime.now(tz=UTC)

    async def persist_authz_denied(
        self, *, request_id: Any, actor_id: Any, voter_votes: Any,
    ) -> AuthzDenial:
        denial = AuthzDenial(
            id=uuid4(),
            request_id=request_id,
            actor_id=actor_id,
            voter_votes=dict(voter_votes),
            attempted_at=datetime.now(tz=UTC),
        )
        self._denials.append(denial)
        return denial

    async def list_pending(self, *, limit: int = 100) -> list[BlockingRequirement]:
        return [
            r for r in self._requests.values()
            if r.status == BlockingRequirementStatus.PENDING
        ][:limit]

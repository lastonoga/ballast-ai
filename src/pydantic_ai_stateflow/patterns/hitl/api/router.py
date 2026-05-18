from __future__ import annotations

from typing import Any
from uuid import UUID

from dbos import DBOS
from fastapi import APIRouter, Header, HTTPException
from pydantic import TypeAdapter

from pydantic_ai_stateflow.patterns.hitl.policy import Policy
from pydantic_ai_stateflow.patterns.hitl.response import HITLResponse
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence import HITLRepository

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


def build_hitl_router(
    *,
    repo: HITLRepository,
    policy: Policy,
    prefix: str = "",
) -> APIRouter:
    """Build a FastAPI router for HITL inbound endpoints.

    Mounts:
      - `POST {prefix}/hitl/{request_id}/respond` — UI / generic JSON.

    Tenant is taken from the `X-Tenant-Id` header (apps wire their own
    tenant resolver via FastAPI middleware that injects the header).

    Authz happens HERE (endpoint side, point #1 of spec 2C.4's two-point
    check). The defense-in-depth check lives in HITLGate.run (SP5).
    """

    router = APIRouter(prefix=prefix)

    async def _respond(
        request_id: UUID, body_json: dict[str, Any], tenant_id: UUID,
    ) -> dict[str, str]:
        request = await repo.load_request(request_id, tenant_id=tenant_id)
        if request is None:
            raise HTTPException(status_code=404, detail="HITL request not found")

        try:
            response = _RESPONSE_ADAPTER.validate_python(body_json)
        except Exception as exc:  # pragma: no cover - pydantic raises ValidationError
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        verdict = await policy.can(
            actor=response.actor_id,
            action="decide",
            resource=request.payload,
            tenant_id=tenant_id,
        )
        if not verdict.is_grant:
            await repo.persist_authz_denied(
                request_id=request_id,
                actor_id=response.actor_id or "<anonymous>",
                voter_votes=dict(verdict.votes),
                tenant_id=tenant_id,
            )
            raise HTTPException(status_code=403, detail=verdict.summary())

        DBOS.send(
            destination_id=str(request.workflow_id),
            message=response.model_dump(mode="json"),
            topic=_hitl_topic(tenant_id, request_id),
        )
        return {"status": "delivered"}

    @router.post("/hitl/{request_id}/respond")
    async def respond_to_hitl(
        request_id: UUID,
        body: dict[str, Any],
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> dict[str, str]:
        if x_tenant_id is None:
            raise HTTPException(status_code=400, detail="X-Tenant-Id header required")
        try:
            tenant_id = UUID(x_tenant_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="X-Tenant-Id must be a UUID",
            ) from exc
        return await _respond(request_id, body, tenant_id)

    router._respond_impl = _respond  # type: ignore[attr-defined]
    return router

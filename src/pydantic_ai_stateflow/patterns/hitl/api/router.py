from __future__ import annotations

import hmac
import json
from hashlib import sha256
from typing import Any
from uuid import UUID

from dbos import DBOS

from pydantic_ai_stateflow.durable import Durable
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import TypeAdapter

from pydantic_ai_stateflow.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
)
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
    webhook_secret: str | None = None,
) -> APIRouter:
    """Build a FastAPI router for HITL inbound endpoints.

    Mounts:
      - ``POST {prefix}/hitl/{request_id}/respond`` — UI / generic JSON.
      - ``POST {prefix}/hitl/webhook/{request_id}`` — signed webhook callback
        (only mounted when ``webhook_secret`` is supplied).

    Authz happens HERE (endpoint side). The defense-in-depth check
    lives in ``HITLGate.run``.
    """

    router = APIRouter(prefix=prefix)

    async def _respond(
        request_id: UUID, body_json: dict[str, Any],
    ) -> dict[str, str]:
        request = await repo.load_request(request_id)
        if request is None:
            raise HTTPException(status_code=404, detail="HITL request not found")

        try:
            response = _RESPONSE_ADAPTER.validate_python(body_json)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        verdict = await policy.can(
            actor=response.actor_id,
            action="decide",
            resource=request.payload,
        )
        if not verdict.is_grant:
            await repo.persist_authz_denied(
                request_id=request_id,
                actor_id=response.actor_id or "<anonymous>",
                voter_votes=dict(verdict.votes),
            )
            raise HTTPException(status_code=403, detail=verdict.summary())

        Durable.send(
            destination_id=str(request.workflow_id),
            message=response.model_dump(mode="json"),
            topic=_hitl_topic(request_id),
        )
        return {"status": "delivered"}

    @router.post("/hitl/{request_id}/respond")
    async def respond_to_hitl(
        request_id: UUID,
        body: dict[str, Any],
    ) -> dict[str, str]:
        return await _respond(request_id, body)

    if webhook_secret is not None:
        secret_bytes = webhook_secret.encode("utf-8")

        @router.post("/hitl/webhook/{request_id}")
        async def respond_via_webhook(
            request_id: UUID,
            request: Request,
            x_stateflow_signature: str | None = Header(
                default=None, alias=WEBHOOK_SIGNATURE_HEADER,
            ),
        ) -> dict[str, str]:
            if x_stateflow_signature is None:
                raise HTTPException(
                    status_code=401, detail="signature header missing",
                )
            raw = await request.body()
            expected = hmac.new(secret_bytes, raw, sha256).hexdigest()
            if not hmac.compare_digest(expected, x_stateflow_signature):
                raise HTTPException(status_code=401, detail="signature mismatch")
            body_json = json.loads(raw.decode("utf-8"))
            return await _respond(request_id, body_json)

    router._respond_impl = _respond  # type: ignore[attr-defined]
    return router

from __future__ import annotations

import hmac
import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, ClassVar, cast
from uuid import UUID

import httpx
from dbos import DBOS
from pydantic import BaseModel, ConfigDict, HttpUrl, TypeAdapter

from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)

WEBHOOK_SIGNATURE_HEADER = "X-Stateflow-Signature"


class WebhookConfig(BaseModel):
    """Outbound webhook configuration: URL to POST to + shared secret for HMAC."""

    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    secret: str


def sign_payload(payload: bytes, *, secret: str) -> str:
    """HMAC-SHA256 of ``payload`` keyed by ``secret``, hex-encoded.

    Verifiers reconstruct the signature with the same secret and compare
    via :func:`hmac.compare_digest` to prevent timing leaks.
    """
    return hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()


@DBOS.step()
async def post_webhook(*, url: str, body: bytes, signature: str) -> None:
    """POST ``body`` to ``url`` with the signature header.

    Wrapped as ``@DBOS.step`` so DBOS records the side-effect (idempotency
    on replay = the step is recorded once even though the HTTP call may
    have been at-least-once at the network level).

    Caller's responsibility:
    - ``body`` must be the exact bytes that were signed (no re-serialization).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            content=body,
            headers={
                WEBHOOK_SIGNATURE_HEADER: signature,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()


class WebhookChannel:
    """Outbound notification + inbound callback HITL channel.

    Flow:
      1. ``ask()`` serializes prompt + request_id into a deterministic JSON
         body, signs it via HMAC-SHA256 with the configured secret, and
         POSTs it to ``config.url`` via :func:`post_webhook` (a DBOS step).
      2. Caller (third party) eventually POSTs a ``HITLResponse`` to
         ``POST /hitl/webhook/{request_id}`` (mounted by ``build_hitl_router``
         when ``webhook_secret`` is supplied). That endpoint verifies the
         signature, runs endpoint-side authz, and ``DBOS.send``s the response
         to the gate's tenant-scoped topic.
      3. ``ask()`` returns via ``DBOS.recv`` on that topic.
    """

    name: ClassVar[str] = "webhook"

    def __init__(self, *, config: WebhookConfig) -> None:
        self.config = config

    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse:
        body = self._build_outbound_body(prompt, request_id)
        signature = sign_payload(body, secret=self.config.secret)
        await post_webhook(url=str(self.config.url), body=body, signature=signature)

        topic = _hitl_topic(prompt.tenant_id, request_id)
        timeout_seconds = (
            prompt.timeout.total_seconds() if prompt.timeout is not None else None
        )
        payload = await DBOS.recv(topic, timeout_seconds=cast(Any, timeout_seconds))
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)

    @staticmethod
    def _build_outbound_body(prompt: HITLPrompt, request_id: UUID) -> bytes:
        payload = {
            "request_id": str(request_id),
            "tenant_id": str(prompt.tenant_id),
            "prompt": prompt.model_dump(mode="json"),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

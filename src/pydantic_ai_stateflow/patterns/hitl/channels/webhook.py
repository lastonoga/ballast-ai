from __future__ import annotations

import hmac
from hashlib import sha256

import httpx
from dbos import DBOS
from pydantic import BaseModel, ConfigDict, HttpUrl

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

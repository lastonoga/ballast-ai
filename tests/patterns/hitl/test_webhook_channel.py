from __future__ import annotations

import hmac
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.patterns.hitl.api.router import build_hitl_router
from pydantic_ai_stateflow.patterns.hitl.channel import HITLChannel
from pydantic_ai_stateflow.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookChannel,
    WebhookConfig,
)
from pydantic_ai_stateflow.patterns.hitl.policy import AllowAll
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    ApprovedResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence import InMemoryHITLRepository


def test_webhook_channel_satisfies_protocol():
    cfg = WebhookConfig(url="https://x.example/cb", secret="s")
    assert isinstance(WebhookChannel(config=cfg), HITLChannel)


@pytest.mark.asyncio
async def test_webhook_channel_posts_signed_payload_then_recvs():
    rid = uuid4()
    prompt = HITLPrompt(
        title="t", context="c",
        decision_kinds={"approved"},
        timeout=timedelta(seconds=5),
    )
    posted: dict = {}

    async def fake_post(*, url: str, body: bytes, signature: str) -> None:
        posted["url"] = url
        posted["body"] = body
        posted["signature"] = signature

    payload = ApprovedResponse(
        actor_id="ext", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)

    cfg = WebhookConfig(url="https://hooks.example/cb", secret="sssh")
    channel = WebhookChannel(config=cfg)
    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.webhook.post_webhook",
        fake_post,
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.webhook.DBOS.recv", recv,
    ):
        result = await channel.ask(prompt, request_id=rid)

    assert isinstance(result, ApprovedResponse)
    assert result.actor_id == "ext"
    body_json = json.loads(posted["body"])
    assert body_json["request_id"] == str(rid)
    assert body_json["prompt"]["title"] == "t"
    expected = hmac.new(b"sssh", posted["body"], sha256).hexdigest()
    assert posted["signature"] == expected
    recv.assert_awaited_once_with(_hitl_topic(rid), timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_webhook_channel_returns_timeout_on_none():
    rid = uuid4()
    prompt = HITLPrompt(
        title="t", context="c",
        decision_kinds={"approved"},
        timeout=timedelta(seconds=1),
    )

    cfg = WebhookConfig(url="https://hooks.example/cb", secret="sssh")
    channel = WebhookChannel(config=cfg)
    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.webhook.post_webhook",
        AsyncMock(return_value=None),
    ), patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.webhook.DBOS.recv",
        AsyncMock(return_value=None),
    ):
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, TimeoutResponse)


@pytest.mark.asyncio
async def test_webhook_endpoint_rejects_missing_signature():
    repo = InMemoryHITLRepository()
    req = await repo.persist_request(
        prompt={"title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval",
    )
    app = FastAPI()
    app.include_router(build_hitl_router(
        repo=repo, policy=AllowAll(), webhook_secret="sssh",
    ))
    body = {"kind": "approved", "actor_id": "ext",
            "answered_at": datetime.now(tz=UTC).isoformat()}
    with TestClient(app) as client:
        r = client.post(f"/hitl/webhook/{req.id}", json=body)
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_webhook_endpoint_rejects_bad_signature():
    repo = InMemoryHITLRepository()
    req = await repo.persist_request(
        prompt={"title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval",
    )
    app = FastAPI()
    app.include_router(build_hitl_router(
        repo=repo, policy=AllowAll(), webhook_secret="sssh",
    ))
    with TestClient(app) as client:
        r = client.post(
            f"/hitl/webhook/{req.id}",
            content=b'{"kind":"approved","actor_id":"ext","answered_at":"2026-01-01T00:00:00+00:00"}',
            headers={
                "Content-Type": "application/json",
                WEBHOOK_SIGNATURE_HEADER: "deadbeef",
            },
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_webhook_endpoint_accepts_valid_signature_and_sends():
    repo = InMemoryHITLRepository()
    wf_id = uuid4()
    req = await repo.persist_request(
        prompt={"title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=wf_id, gate_kind="hitl_gate",
        purpose="approval",
    )
    app = FastAPI()
    app.include_router(build_hitl_router(
        repo=repo, policy=AllowAll(), webhook_secret="sssh",
    ))

    body = b'{"actor_id":"ext","answered_at":"2026-01-01T00:00:00+00:00","kind":"approved"}'
    sig = hmac.new(b"sssh", body, sha256).hexdigest()
    sent: dict = {}

    def fake_send(destination_id, message, topic=None):
        sent.update(destination_id=destination_id, message=message, topic=topic)

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.api.router.DBOS.send", fake_send,
    ), TestClient(app) as client:
        r = client.post(
            f"/hitl/webhook/{req.id}",
            content=body,
            headers={
                "Content-Type": "application/json",
                WEBHOOK_SIGNATURE_HEADER: sig,
            },
        )
    assert r.status_code == 200
    assert sent["destination_id"] == str(wf_id)
    assert sent["topic"] == _hitl_topic(req.id)


@pytest.mark.asyncio
async def test_webhook_endpoint_404_when_no_secret_configured():
    """If `webhook_secret` not provided, the webhook endpoint MUST NOT mount."""
    repo = InMemoryHITLRepository()
    app = FastAPI()
    app.include_router(build_hitl_router(repo=repo, policy=AllowAll()))
    with TestClient(app) as client:
        r = client.post(f"/hitl/webhook/{uuid4()}", json={})
    assert r.status_code == 404

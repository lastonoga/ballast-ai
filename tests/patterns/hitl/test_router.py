from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.patterns.hitl.api.router import build_hitl_router
from ballast.patterns.hitl.policy import AllowAll, DenyAll, Policy
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence import (
    HITLRepository,
    InMemoryHITLRepository,
)


def _make_app(repo: HITLRepository, policy: Policy) -> FastAPI:
    app = FastAPI()
    app.include_router(build_hitl_router(repo=repo, policy=policy))
    return app


@pytest.mark.asyncio
async def test_respond_404_when_request_unknown() -> None:
    repo = InMemoryHITLRepository()
    app = _make_app(repo, AllowAll())
    body = {
        "kind": "approved",
        "actor_id": "alice",
        "answered_at": datetime.now(tz=UTC).isoformat(),
    }
    with TestClient(app) as client:
        r = client.post(f"/hitl/{uuid4()}/respond", json=body)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_respond_403_when_policy_denies_and_audited() -> None:
    repo = InMemoryHITLRepository()
    req = await repo.persist_request(
        prompt={"title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval",
    )
    app = _make_app(repo, DenyAll())
    body = {
        "kind": "approved", "actor_id": "alice",
        "answered_at": datetime.now(tz=UTC).isoformat(),
    }
    with TestClient(app) as client:
        r = client.post(f"/hitl/{req.id}/respond", json=body)
    assert r.status_code == 403
    assert len(repo._denials) == 1
    assert repo._denials[0].actor_id == "alice"


@pytest.mark.asyncio
async def test_respond_200_sends_to_topic_on_grant() -> None:
    repo = InMemoryHITLRepository()
    wf_id = uuid4()
    req = await repo.persist_request(
        prompt={"title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=wf_id, gate_kind="hitl_gate",
        purpose="approval",
    )
    app = _make_app(repo, AllowAll())
    body = {
        "kind": "approved", "actor_id": "alice",
        "answered_at": datetime.now(tz=UTC).isoformat(),
    }
    sent: dict[str, Any] = {}

    def fake_send(
        destination_id: str, message: Any, topic: str | None = None,
    ) -> None:
        sent["destination_id"] = destination_id
        sent["message"] = message
        sent["topic"] = topic

    with (
        patch("ballast.patterns.hitl.api.router.DBOS.send", fake_send),
        TestClient(app) as client,
    ):
        r = client.post(f"/hitl/{req.id}/respond", json=body)
    assert r.status_code == 200
    assert sent["destination_id"] == str(wf_id)
    assert sent["topic"] == _hitl_topic(req.id)
    assert sent["message"]["kind"] == "approved"
    assert sent["message"]["actor_id"] == "alice"


@pytest.mark.asyncio
async def test_router_uses_provided_path_prefix() -> None:
    repo = InMemoryHITLRepository()
    app = FastAPI()
    app.include_router(
        build_hitl_router(repo=repo, policy=AllowAll(), prefix="/api"),
    )
    with TestClient(app) as client:
        r = client.post(
            f"/api/hitl/{uuid4()}/respond",
            json={
                "kind": "approved",
                "answered_at": datetime.now(tz=UTC).isoformat(),
            },
        )
    assert r.status_code == 404

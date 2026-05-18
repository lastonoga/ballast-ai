from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.a2a import (
    AgentCard,
    build_a2a_router,
)


class _EchoAgent:
    """Minimal A2AAgentAdapter for tests."""

    name = "echo"
    description = "echoes the last message"

    async def run(
        self, *, messages: list[dict[str, Any]], tenant_id: UUID,
    ) -> dict[str, Any]:
        return {"echo": messages[-1] if messages else None, "tenant": str(tenant_id)}


def test_well_known_agent_json_returns_cards() -> None:
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    with TestClient(app) as c:
        r = c.get("/.well-known/agent.json")
    assert r.status_code == 200
    body = r.json()
    assert "agents" in body
    assert any(card["name"] == "echo" for card in body["agents"])


def test_well_known_agent_json_card_includes_endpoint() -> None:
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    with TestClient(app) as c:
        r = c.get("/.well-known/agent.json")
    card = next(c for c in r.json()["agents"] if c["name"] == "echo")
    assert card["endpoint"].endswith("/a2a/echo")


@pytest.mark.asyncio
async def test_a2a_invoke_routes_to_agent() -> None:
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    tid = uuid4()
    body = {"messages": [{"role": "user", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            "/a2a/echo", json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    payload = r.json()
    assert payload["echo"]["text"] == "hi"
    assert payload["tenant"] == str(tid)


@pytest.mark.asyncio
async def test_a2a_invoke_404_when_unknown_agent() -> None:
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    with TestClient(app) as c:
        r = c.post(
            "/a2a/ghost",
            json={"messages": []},
            headers={"X-Tenant-Id": str(uuid4())},
        )
    assert r.status_code == 404


def test_agent_card_includes_optional_metadata() -> None:
    """Cards carry capabilities + description so discovery is useful."""
    card = AgentCard(
        name="planner", description="plans things",
        endpoint="/a2a/planner",
        capabilities=["plan", "decompose"],
    )
    assert "plan" in card.capabilities
    assert card.description == "plans things"

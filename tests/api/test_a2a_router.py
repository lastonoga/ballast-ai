from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.a2a import (
    AgentCard,
    build_a2a_router,
)
from ballast.api.error_middleware import stateflow_error_handler
from ballast.errors import BallastError


def _install_error_handler(app: FastAPI) -> None:
    app.add_exception_handler(BallastError, stateflow_error_handler)


class _EchoAgent:
    """Minimal A2AAgentAdapter for tests."""

    name = "echo"
    description = "echoes the last message"

    async def run(self, *, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"echo": messages[-1] if messages else None}


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
    body = {"messages": [{"role": "user", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post("/a2a/echo", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["echo"]["text"] == "hi"


@pytest.mark.asyncio
async def test_a2a_invoke_404_when_unknown_agent() -> None:
    app = FastAPI()
    _install_error_handler(app)
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    with TestClient(app) as c:
        r = c.post("/a2a/ghost", json={"messages": []})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "BALLAST_AGENT_NOT_REGISTERED"


def test_agent_card_includes_optional_metadata() -> None:
    """Cards carry capabilities + description so discovery is useful."""
    card = AgentCard(
        name="planner", description="plans things",
        endpoint="/a2a/planner",
        capabilities=["plan", "decompose"],
    )
    assert "plan" in card.capabilities
    assert card.description == "plans things"

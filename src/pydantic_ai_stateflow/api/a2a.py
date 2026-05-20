from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


@runtime_checkable
class A2AAgentAdapter(Protocol):
    """Minimal contract for an agent exposed over A2A.

    Apps that need per-call scoping (tenant, user) carry it inside
    ``messages`` payload or via their adapter's constructor state —
    the framework's A2A surface is identity-agnostic.
    """

    name: str
    description: str

    async def run(
        self, *, messages: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class AgentCard(BaseModel):
    name: str
    description: str
    endpoint: str
    capabilities: list[str] = Field(default_factory=list)


class _InvokeBody(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)


def build_a2a_router(
    *,
    agents: dict[str, A2AAgentAdapter],
    prefix: str = "",
) -> APIRouter:
    """Mount A2A discovery (``/.well-known/agent.json``) + invoke (``/a2a/{name}``)."""
    router = APIRouter(prefix=prefix)
    registry = dict(agents)

    @router.get("/.well-known/agent.json")
    async def agent_cards() -> dict[str, Any]:
        cards = [
            AgentCard(
                name=name,
                description=getattr(adapter, "description", ""),
                endpoint=f"{prefix}/a2a/{name}",
                capabilities=list(getattr(adapter, "capabilities", [])),
            ).model_dump()
            for name, adapter in registry.items()
        ]
        return {"agents": cards}

    @router.post("/a2a/{agent_name}")
    async def invoke(
        agent_name: str,
        body: _InvokeBody,
    ) -> dict[str, Any]:
        adapter = registry.get(agent_name)
        if adapter is None:
            raise HTTPException(
                status_code=404, detail=f"unknown agent: {agent_name}",
            )
        return await adapter.run(messages=body.messages)

    return router

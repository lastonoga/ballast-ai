from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from pydantic_ai_stateflow.api.deps import get_tenant_id
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

_TenantDep = Depends(get_tenant_id)


class CreateThreadBody(BaseModel):
    purpose: str
    purpose_metadata: dict[str, Any] = Field(default_factory=dict)
    actor_id: str


def build_threads_router(
    *,
    thread_repo: ThreadRepository,
    prefix: str = "",
) -> APIRouter:
    """REST surface for the Thread aggregate (SP2)."""
    router = APIRouter(prefix=prefix)

    @router.post("/threads", status_code=201)
    async def create_thread(
        body: CreateThreadBody,
        tenant_id: UUID = _TenantDep,
    ) -> dict[str, Any]:
        thread = await thread_repo.create(
            purpose=body.purpose,
            purpose_metadata=body.purpose_metadata,
            actor_id=body.actor_id,
            tenant_id=tenant_id,
        )
        return thread.model_dump(mode="json")

    @router.get("/threads/{thread_id}")
    async def get_thread(
        thread_id: UUID,
        tenant_id: UUID = _TenantDep,
    ) -> dict[str, Any]:
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return thread.model_dump(mode="json")

    @router.get("/threads/{thread_id}/messages")
    async def get_messages(
        thread_id: UUID,
        tenant_id: UUID = _TenantDep,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        msgs = await thread_repo.history(
            thread_id, tenant_id=tenant_id, limit=limit,
        )
        return [m.model_dump(mode="json") for m in msgs]

    return router

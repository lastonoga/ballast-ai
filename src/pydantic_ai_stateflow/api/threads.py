from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from pydantic_ai_stateflow.api.deps import get_tenant_id
from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

_log = get_logger(__name__)

_TenantDep = Depends(get_tenant_id)
_IncludeArchivedQuery = Query(default=False)
_LimitQuery = Query(default=100, ge=1, le=500)
_OffsetQuery = Query(default=0, ge=0)


class CreateThreadBody(BaseModel):
    purpose: str
    purpose_metadata: dict[str, Any] = Field(default_factory=dict)
    actor_id: str


class _RenameBody(BaseModel):
    title: str | None = None


def build_threads_router(
    *,
    thread_repo: ThreadRepository,
    prefix: str = "",
) -> APIRouter:
    """REST surface for the Thread aggregate (SP2 + F6).

    Implements the CRUD shape required by assistant-ui's
    ``RemoteThreadListAdapter`` contract:

    - ``POST   /threads``               → ``initialize`` (create)
    - ``GET    /threads``               → ``list`` (newest-first, archive-filtered)
    - ``GET    /threads/{id}``          → load one
    - ``GET    /threads/{id}/messages`` → ``history``
    - ``PATCH  /threads/{id}``          → ``rename`` (title)
    - ``POST   /threads/{id}/archive``  → ``archive``
    - ``POST   /threads/{id}/unarchive``→ unarchive
    - ``DELETE /threads/{id}``          → ``delete`` (idempotent)

    Agent-driven ``generateTitle`` (streaming the title from an LLM) is
    deferred: apps can summarize the thread themselves and call ``PATCH``
    with the resulting string. The framework only exposes the persistence
    primitive here.
    """
    router = APIRouter(prefix=prefix)

    @router.post("/threads", status_code=201)
    @traced(
        TraceName.THREADS_CREATE,
        attrs=lambda body, tenant_id=None, **__: {
            "tenant_id": str(tenant_id) if tenant_id else "<dep>",
            "purpose": body.purpose,
        },
    )
    async def create_thread(
        body: CreateThreadBody,
        tenant_id: UUID = _TenantDep,
    ) -> dict[str, Any]:
        _log.info(
            "POST /threads (tenant=%s purpose=%s actor=%s)",
            tenant_id, body.purpose, body.actor_id,
        )
        thread = await thread_repo.create(
            purpose=body.purpose,
            purpose_metadata=body.purpose_metadata,
            actor_id=body.actor_id,
            tenant_id=tenant_id,
        )
        return thread.model_dump(mode="json")

    @router.get("/threads")
    async def list_threads(
        tenant_id: UUID = _TenantDep,
        include_archived: bool = _IncludeArchivedQuery,
        limit: int = _LimitQuery,
        offset: int = _OffsetQuery,
    ) -> list[dict[str, Any]]:
        threads = await thread_repo.list_(
            tenant_id=tenant_id,
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        )
        return [t.model_dump(mode="json") for t in threads]

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
    @traced(
        TraceName.THREADS_GET_MESSAGES,
        attrs=lambda thread_id, tenant_id=None, limit=100, **__: {
            "thread_id": str(thread_id),
            "tenant_id": str(tenant_id) if tenant_id else "<dep>",
            "limit": limit,
        },
    )
    async def get_messages(
        thread_id: UUID,
        tenant_id: UUID = _TenantDep,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        _log.debug(
            "GET /threads/%s/messages (tenant=%s limit=%d)",
            thread_id, tenant_id, limit,
        )
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            _log.warning(
                "GET /threads/%s/messages → 404 (tenant=%s)",
                thread_id, tenant_id,
            )
            raise HTTPException(status_code=404, detail="thread not found")
        msgs = await thread_repo.history(
            thread_id, tenant_id=tenant_id, limit=limit,
        )
        _log.debug(
            "GET /threads/%s/messages → %d msgs", thread_id, len(msgs),
        )
        return [m.model_dump(mode="json") for m in msgs]

    @router.patch("/threads/{thread_id}")
    async def rename_thread(
        thread_id: UUID,
        body: _RenameBody,
        tenant_id: UUID = _TenantDep,
    ) -> dict[str, Any]:
        try:
            thread = await thread_repo.rename(
                thread_id, title=body.title, tenant_id=tenant_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="thread not found") from exc
        return thread.model_dump(mode="json")

    @router.post("/threads/{thread_id}/archive")
    async def archive_thread(
        thread_id: UUID,
        tenant_id: UUID = _TenantDep,
    ) -> dict[str, Any]:
        try:
            thread = await thread_repo.archive(thread_id, tenant_id=tenant_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="thread not found") from exc
        return thread.model_dump(mode="json")

    @router.post("/threads/{thread_id}/unarchive")
    async def unarchive_thread(
        thread_id: UUID,
        tenant_id: UUID = _TenantDep,
    ) -> dict[str, Any]:
        try:
            thread = await thread_repo.unarchive(thread_id, tenant_id=tenant_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="thread not found") from exc
        return thread.model_dump(mode="json")

    @router.delete("/threads/{thread_id}", status_code=204)
    async def delete_thread(
        thread_id: UUID,
        tenant_id: UUID = _TenantDep,
    ) -> Response:
        await thread_repo.delete(thread_id, tenant_id=tenant_id)
        return Response(status_code=204)

    return router

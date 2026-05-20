from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

_log = get_logger(__name__)

_IncludeArchivedQuery = Query(default=False)
_LimitQuery = Query(default=100, ge=1, le=500)
_OffsetQuery = Query(default=0, ge=0)


def build_threads_router(
    *,
    thread_repo: ThreadRepository,
    prefix: str = "",
) -> APIRouter:
    """REST surface for the Thread aggregate — READ / lifecycle / DELETE.

    - ``GET    /threads``               → list (newest-first, archive-filtered)
    - ``GET    /threads/{id}``          → load one
    - ``GET    /threads/{id}/messages`` → history
    - ``POST   /threads/{id}/archive``  → archive
    - ``POST   /threads/{id}/unarchive``→ unarchive
    - ``DELETE /threads/{id}``          → delete (idempotent)

    **Thread creation is the app's responsibility** — apps write a
    custom ``POST /threads`` calling ``repo.create(agent=..., metadata=...)``.

    **No title / rename endpoint.** Apps store title in
    ``thread.metadata`` and expose their own PATCH that calls
    ``repo.update_metadata(thread_id, metadata={...})``.

    **No tenant / actor filtering at this layer.** Framework repo is
    scoping-unaware; apps wrap / compose for multi-tenancy.
    """
    router = APIRouter(prefix=prefix)

    @router.get("/threads")
    async def list_threads(
        include_archived: bool = _IncludeArchivedQuery,
        limit: int = _LimitQuery,
        offset: int = _OffsetQuery,
    ) -> list[dict[str, Any]]:
        threads = await thread_repo.list_(
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        )
        return [t.model_dump(mode="json", by_alias=True) for t in threads]

    @router.get("/threads/{thread_id}")
    async def get_thread(thread_id: UUID) -> dict[str, Any]:
        thread = await thread_repo.load(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return thread.model_dump(mode="json", by_alias=True)

    @router.get("/threads/{thread_id}/messages")
    @traced(
        TraceName.THREADS_GET_MESSAGES,
        attrs=lambda thread_id, limit=100, **__: {
            "thread_id": str(thread_id),
            "limit": limit,
        },
    )
    async def get_messages(
        thread_id: UUID, limit: int = 100,
    ) -> list[dict[str, Any]]:
        thread = await thread_repo.load(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        msgs = await thread_repo.history(thread_id, limit=limit)
        return [m.model_dump(mode="json") for m in msgs]

    @router.post("/threads/{thread_id}/archive")
    async def archive_thread(thread_id: UUID) -> dict[str, Any]:
        try:
            thread = await thread_repo.archive(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="thread not found") from exc
        return thread.model_dump(mode="json", by_alias=True)

    @router.post("/threads/{thread_id}/unarchive")
    async def unarchive_thread(thread_id: UUID) -> dict[str, Any]:
        try:
            thread = await thread_repo.unarchive(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="thread not found") from exc
        return thread.model_dump(mode="json", by_alias=True)

    @router.delete("/threads/{thread_id}", status_code=204)
    async def delete_thread(thread_id: UUID) -> Response:
        await thread_repo.delete(thread_id)
        return Response(status_code=204)

    return router

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, Response

from pydantic_ai_stateflow.errors import ThreadNotFound
from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

_log = get_logger(__name__)

_IncludeArchivedQuery = Query(default=False)
_LimitQuery = Query(default=100, ge=1, le=500)
_OffsetQuery = Query(default=0, ge=0)


# ── Module-level router ──────────────────────────────────────────────
#
# REST surface for the Thread aggregate — READ / lifecycle / DELETE.
# Resolves ``thread_repo`` via ``Depends(get_thread_repo)`` from
# ``app.state``. ``sf.create_app()`` mounts this router.

from fastapi import Depends as _Depends  # noqa: E402

from pydantic_ai_stateflow.api.deps import get_thread_repo as _get_thread_repo  # noqa: E402

threads_router = APIRouter()


@threads_router.get("/threads")
async def _list_threads(
    include_archived: bool = _IncludeArchivedQuery,
    limit: int = _LimitQuery,
    offset: int = _OffsetQuery,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
) -> list[dict[str, Any]]:
    threads = await thread_repo.list_(
        include_archived=include_archived,
        limit=limit,
        offset=offset,
    )
    return [t.model_dump(mode="json", by_alias=True) for t in threads]


@threads_router.get("/threads/{thread_id}")
async def _get_thread(
    thread_id: UUID,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
) -> dict[str, Any]:
    thread = await thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(thread_id=str(thread_id))
    return thread.model_dump(mode="json", by_alias=True)


@threads_router.get("/threads/{thread_id}/messages")
@traced(
    TraceName.THREADS_GET_MESSAGES,
    attrs=lambda thread_id, limit=1000, **__: {
        "thread_id": str(thread_id),
        "limit": limit,
    },
)
async def _get_messages(
    thread_id: UUID,
    limit: int = 1000,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
) -> list[dict[str, Any]]:
    """Return the thread's linear message list, ordered by ``created_at``."""
    thread = await thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(thread_id=str(thread_id))
    msgs = await thread_repo.history(thread_id, limit=limit)
    return [m.model_dump(mode="json") for m in msgs]


@threads_router.post("/threads/{thread_id}/archive")
async def _archive_thread(
    thread_id: UUID,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
) -> dict[str, Any]:
    try:
        thread = await thread_repo.archive(thread_id)
    except KeyError as exc:
        raise ThreadNotFound(thread_id=str(thread_id)) from exc
    return thread.model_dump(mode="json", by_alias=True)


@threads_router.post("/threads/{thread_id}/unarchive")
async def _unarchive_thread(
    thread_id: UUID,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
) -> dict[str, Any]:
    try:
        thread = await thread_repo.unarchive(thread_id)
    except KeyError as exc:
        raise ThreadNotFound(thread_id=str(thread_id)) from exc
    return thread.model_dump(mode="json", by_alias=True)


@threads_router.delete("/threads/{thread_id}", status_code=204)
async def _delete_thread(
    thread_id: UUID,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
) -> Response:
    await thread_repo.delete(thread_id)
    return Response(status_code=204)

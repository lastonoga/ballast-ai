from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse

from ballast.errors import ThreadNotFound
from ballast.logging import get_logger
from ballast.observability.spans import traced
from ballast.observability.trace_names import TraceName
from ballast.persistence.events.repository import EventLogRepository
from ballast.persistence.thread.repository import ThreadRepository
from ballast.runtime.event_stream import EventStream, thread_channel

_log = get_logger(__name__)

_IncludeArchivedQuery = Query(default=False)
_LimitQuery = Query(default=100, ge=1, le=500)
_OffsetQuery = Query(default=0, ge=0)


# ── Module-level router ──────────────────────────────────────────────
#
# REST surface for the Thread aggregate — READ / lifecycle / DELETE.
# Resolves ``thread_repo`` via ``Depends(get_thread_repo)`` from
# ``app.state``. ``ballast.create_app()`` mounts this router.

from fastapi import Depends as _Depends  # noqa: E402

from ballast.api.deps import (  # noqa: E402
    get_event_log as _get_event_log,
    get_event_stream as _get_event_stream,
    get_thread_repo as _get_thread_repo,
)

threads_router = APIRouter()

# Period for SSE keep-alive comments. Browsers + intermediate proxies
# happily drop idle connections after ~30s — 25s gives some headroom.
_SSE_KEEPALIVE_SECONDS = 25.0


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


@threads_router.get("/threads/{thread_id}/events")
async def _thread_events(
    request: Request,
    thread_id: UUID,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
    event_log: EventLogRepository = _Depends(_get_event_log),
    event_stream: EventStream = _Depends(_get_event_stream),
) -> StreamingResponse:
    """Server-Sent Events stream of the thread's event log.

    Replays missed events since ``Last-Event-ID`` (numeric ``seq`` of
    the last delivered event), then tails ``event_stream`` for live
    notifications. Each SSE message carries ``{"kind", "payload"}``
    mirroring the log's row shape so frontends can dispatch on
    ``kind`` (``message-added``, ``thread-created``, …) without
    polling.

    The handler keeps the connection alive with a ``:keep-alive``
    comment every ~25s; closes on client disconnect; never holds
    write locks (notifications go through an in-memory queue +
    ``event_log.read_since`` for content).
    """
    thread = await thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(thread_id=str(thread_id))

    try:
        last_seq = int(last_event_id) if last_event_id else 0
    except (TypeError, ValueError):
        last_seq = 0

    async def gen() -> Any:
        cursor = last_seq

        # 1. Replay anything missed during disconnect (or the full
        #    history on first connect with no Last-Event-ID).
        missed = await event_log.read_since(thread_id, after_seq=cursor)
        for ev in missed:
            cursor = ev.seq
            yield _sse_pack(ev.seq, ev.kind, ev.payload)

        # 2. Subscribe + tail live. We pump notifications into a local
        #    queue from a dedicated reader task so the main loop can
        #    use ``wait_for`` for the keep-alive cadence without
        #    cancelling the underlying subscribe generator (which
        #    would close it permanently).
        async with event_stream.subscribe(
            thread_channel(thread_id),
        ) as notifications:
            local: asyncio.Queue[None] = asyncio.Queue()

            async def reader() -> None:
                try:
                    async for _ in notifications:
                        await local.put(None)
                except asyncio.CancelledError:
                    pass

            reader_task = asyncio.create_task(reader())
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        await asyncio.wait_for(
                            local.get(), timeout=_SSE_KEEPALIVE_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        yield b": keep-alive\n\n"
                        continue
                    # Read all new events since cursor — pick up any
                    # that arrived between the notification fire and
                    # our log read.
                    fresh = await event_log.read_since(
                        thread_id, after_seq=cursor,
                    )
                    for ev in fresh:
                        cursor = ev.seq
                        yield _sse_pack(ev.seq, ev.kind, ev.payload)
            finally:
                reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reader_task

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse_pack(seq: int, kind: str, payload: dict[str, Any]) -> bytes:
    """Encode one event log row as an SSE ``data:`` frame.

    Frontends parse ``ev.data`` as JSON and dispatch on ``kind`` (see
    ``examples/notes-app/frontend/.../runtime-provider.tsx``)."""
    data = json.dumps({"kind": kind, "payload": payload}, default=str)
    return f"id: {seq}\ndata: {data}\n\n".encode()

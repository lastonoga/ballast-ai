"""``ThreadEpisodicSource`` — turns existing thread_repo history into Episodes."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import DetailLevel
from ballast.memory.episodic.sources import ThreadEpisodicSource


class _FakeThreadRepo:
    """Stub with the real list_() + history() signatures."""

    def __init__(self, threads, messages):
        self._threads, self._messages = threads, messages

    async def list_(self, *, include_archived=False, limit=100, offset=0):
        return self._threads[:limit]

    async def history(self, thread_id, *, limit=1000):
        return [m for m in self._messages if m["thread_id"] == thread_id]


@pytest.mark.asyncio
async def test_thread_source_recall_returns_recent_first() -> None:
    now = datetime.now(UTC)
    tid_old, tid_new = "t-old", "t-new"
    repo = _FakeThreadRepo(
        threads=[
            # repo.list_() returns newest-first per its order_by clause
            type("T", (), {"id": tid_new, "created_at": now, "metadata_": {"user_id": "u-1"}})(),
            type("T", (), {"id": tid_old, "created_at": now - timedelta(days=7), "metadata_": {"user_id": "u-1"}})(),
        ],
        messages=[
            {"thread_id": tid_old, "role": "user", "text": "old prompt",
             "created_at": now - timedelta(days=7)},
            {"thread_id": tid_new, "role": "user", "text": "new prompt",
             "created_at": now},
            {"thread_id": tid_new, "role": "assistant", "text": "new reply",
             "created_at": now},
        ],
    )
    src = ThreadEpisodicSource(thread_repo=repo)
    out = await src.recall(
        intent="x", scope=Scope(user_id="u-1"), k=10, detail=DetailLevel.PREVIEW,
    )
    assert len(out) == 2
    assert out[0].episode.id.endswith("t-new")
    assert out[1].episode.id.endswith("t-old")


@pytest.mark.asyncio
async def test_thread_source_filters_by_user_id() -> None:
    now = datetime.now(UTC)
    repo = _FakeThreadRepo(
        threads=[
            type("T", (), {"id": "t-1", "created_at": now, "metadata_": {"user_id": "u-1"}})(),
            type("T", (), {"id": "t-2", "created_at": now, "metadata_": {"user_id": "u-2"}})(),
        ],
        messages=[
            {"thread_id": "t-1", "role": "user", "text": "u1", "created_at": now},
            {"thread_id": "t-2", "role": "user", "text": "u2", "created_at": now},
        ],
    )
    src = ThreadEpisodicSource(thread_repo=repo)
    out = await src.recall(
        intent="x", scope=Scope(user_id="u-1"), k=10, detail=DetailLevel.PREVIEW,
    )
    assert len(out) == 1
    assert out[0].episode.id.endswith("t-1")


@pytest.mark.asyncio
async def test_thread_source_remember_not_supported() -> None:
    from ballast.memory.episodic import Episode

    src = ThreadEpisodicSource(thread_repo=_FakeThreadRepo([], []))
    ep = Episode(id="x", source="thread", occurred_at=datetime.now(UTC),
                 scope=Scope(), preview="p")
    with pytest.raises(NotImplementedError):
        await src.remember(ep)

"""``ThreadEpisodicSource`` — wraps existing thread_repo as episodic memory.

v1 returns recent threads sorted by created_at desc; filters by
``scope.user_id`` against ``thread.metadata_.get('user_id')`` if set.
Phase 1.5 adds a vector index over turn previews if recency proves
insufficient.
"""
from __future__ import annotations

from typing import Any, Protocol

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, Episode, ScoredEpisode,
)


class _ThreadRepo(Protocol):
    async def list_(self, *, include_archived: bool = ..., limit: int = ..., offset: int = ...) -> list[Any]: ...
    async def history(self, thread_id: Any, *, limit: int = ...) -> list[Any]: ...


def _get(o: Any, k: str, default: Any = None) -> Any:
    if hasattr(o, k): return getattr(o, k)
    if isinstance(o, dict): return o.get(k, default)
    return default


class ThreadEpisodicSource:
    """Episodic source backed by the thread repository."""

    name = "thread"

    def __init__(self, *, thread_repo: _ThreadRepo, list_limit: int = 100) -> None:
        self._repo = thread_repo
        self._list_limit = list_limit

    async def recall(
        self, *, intent: str, scope: Scope, k: int, detail: DetailLevel,
    ) -> list[ScoredEpisode]:
        threads = await self._repo.list_(limit=self._list_limit)
        # Filter by user_id stored in thread.metadata_ (if scope sets one).
        scope_user = getattr(scope, "user_id", None)
        if scope_user is not None:
            threads = [
                t for t in threads
                if (_get(t, "metadata_") or {}).get("user_id") == scope_user
            ]
        # Sort newest first (repo already does, but be defensive).
        threads = sorted(threads, key=lambda t: _get(t, "created_at"), reverse=True)
        threads = threads[:k]

        out: list[ScoredEpisode] = []
        for t in threads:
            messages = await self._repo.history(_get(t, "id"))
            if not messages: continue
            user_msg = next(
                (m for m in messages if _get(m, "role") == "user"),
                messages[0],
            )
            preview = (_get(user_msg, "text") or "")[:200]
            summary = None
            if detail >= DetailLevel.SUMMARY:
                assistant_msg = next(
                    (m for m in messages if _get(m, "role") == "assistant"), None,
                )
                summary = preview + (
                    "\n→ " + (_get(assistant_msg, "text") or "")[:300]
                    if assistant_msg else ""
                )
            full = None
            if detail >= DetailLevel.FULL:
                full = {"messages": messages}
            out.append(ScoredEpisode(
                episode=Episode(
                    id=f"thread:{_get(t, 'id')}",
                    source=self.name,
                    occurred_at=_get(t, "created_at"),
                    scope=scope,
                    preview=preview,
                    summary=summary,
                    full=full,
                    references=[],   # v1: no ref extraction
                ),
                score=1.0,            # recency-only — uniform score
            ))
        return out

    async def hydrate(self, episode: Episode, *, detail: DetailLevel) -> Episode:
        if not episode.id.startswith("thread:"):
            raise ValueError(f"{self.name} cannot hydrate id={episode.id}")
        if detail < DetailLevel.SUMMARY: return episode
        thread_id = episode.id.removeprefix("thread:")
        messages = await self._repo.history(thread_id)
        full = {"messages": messages} if detail >= DetailLevel.FULL else episode.full
        return episode.model_copy(update={
            "summary": episode.summary or (
                (_get(messages[0], "text") or "")[:300] if messages else ""
            ),
            "full": full,
        })

    async def remember(self, episode: Episode) -> None:
        raise NotImplementedError(
            "ThreadEpisodicSource is read-only — thread_repo is source-of-truth",
        )


__all__ = ["ThreadEpisodicSource"]

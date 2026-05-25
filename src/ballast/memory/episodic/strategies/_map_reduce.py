"""``MapReduce`` strategy — LLM-driven digest of large recall sets.

Builds on ``ballast.patterns.map_reduce.map_reduce_llm``: per-episode
``map_fn`` runs in parallel (typically an LLM call summarizing one
episode); ``reduce_fn`` synthesizes the final digest. The strategy
wraps the digest in a single synthetic Episode whose ``preview`` is
the digest text — preserving the RecallResult contract.

Apps that want structured digest output should make ``reduce_fn``
return a string and pass that to a follow-on agent run.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, Episode, RecallResult, ScoredEpisode,
)
from ballast.memory.episodic._protocol import EpisodicSource
from ballast.patterns.map_reduce import map_reduce_llm

_log = logging.getLogger(__name__)


class MapReduce:
    """Federate → top max_items → LLM map+reduce → synthetic Episode."""

    requires_grounding = False

    def __init__(
        self,
        *,
        max_items: int,
        map_fn: Callable[[ScoredEpisode], Awaitable[str]],
        reduce_fn: Callable[[list[str]], Awaitable[str]],
        detail: DetailLevel = DetailLevel.FULL,
        map_concurrency: int = 8,
    ) -> None:
        self._max = max_items
        self._map_fn = map_fn
        self._reduce_fn = reduce_fn
        self._detail = detail
        self._map_concurrency = map_concurrency

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe(src):
            try:
                return await src.recall(
                    intent=intent, scope=scope, k=self._max,
                    detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return []
        per_source = await asyncio.gather(*(_safe(s) for s in sources))
        flat: list[ScoredEpisode] = [se for batch in per_source for se in batch][: self._max]
        if not flat:
            return RecallResult(episodes=[])
        digest = await map_reduce_llm(
            items=flat,
            map_step=self._map_fn,
            reduce_step=self._reduce_fn,
            map_concurrency=self._map_concurrency,
        )
        synthesized = Episode(
            id=f"digest:{intent[:32]}",
            source="map-reduce-strategy",
            occurred_at=datetime.now(timezone.utc),
            scope=scope,
            preview=digest,
            summary=digest,
            references=[r for se in flat for r in se.episode.references],
            metadata={"intent": intent, "source_episode_count": len(flat)},
        )
        return RecallResult(episodes=[ScoredEpisode(episode=synthesized, score=1.0)])


__all__ = ["MapReduce"]

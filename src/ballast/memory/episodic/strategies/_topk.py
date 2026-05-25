"""``TopK`` recall strategy — classic RAG default."""
from __future__ import annotations

import asyncio
import logging

from ballast.memory._scope import Scope
from ballast.memory.episodic._mergers import RRFMerger, ScoreMerger
from ballast.memory.episodic._models import DetailLevel, RecallResult
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


class TopK:
    """Query all sources in parallel, merge, return top K."""

    requires_grounding = False

    def __init__(
        self,
        *,
        k: int = 5,
        detail: DetailLevel = DetailLevel.SUMMARY,
        merger: ScoreMerger | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"TopK k must be >= 1, got {k!r}")
        self._k = k
        self._detail = detail
        self._merger = merger or RRFMerger()

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe_recall(src):
            try:
                return src.name, await src.recall(
                    intent=intent, scope=scope, k=self._k, detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return src.name, []

        results = await asyncio.gather(*(_safe_recall(s) for s in sources))
        by_source = {name: episodes for name, episodes in results}
        merged = self._merger.merge(by_source)
        return RecallResult(episodes=merged[: self._k])


__all__ = ["TopK"]

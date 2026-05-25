"""``AllRelevant`` — return all matches above a score threshold."""
from __future__ import annotations

import asyncio
import logging

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import DetailLevel, RecallResult, ScoredEpisode
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


class AllRelevant:
    """Return everything above a threshold — for when context fits."""

    requires_grounding = False

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        detail: DetailLevel = DetailLevel.PREVIEW,
        per_source_limit: int = 100,
    ) -> None:
        self._threshold = threshold
        self._detail = detail
        self._per_source = per_source_limit

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe(src):
            try:
                return await src.recall(
                    intent=intent, scope=scope, k=self._per_source,
                    detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return []

        per = await asyncio.gather(*(_safe(s) for s in sources))
        flat = [se for batch in per for se in batch]
        # Dedup by id, keep highest raw score per id.
        best: dict[str, ScoredEpisode] = {}
        for se in flat:
            if se.episode.id not in best or se.score > best[se.episode.id].score:
                best[se.episode.id] = se
        filtered = [se for se in best.values() if se.score >= self._threshold]
        filtered.sort(key=lambda se: se.score, reverse=True)
        return RecallResult(episodes=filtered)


__all__ = ["AllRelevant"]

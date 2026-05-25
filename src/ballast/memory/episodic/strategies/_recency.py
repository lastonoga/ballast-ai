"""``Recency`` — most-recent N episodes; scores ignored."""
from __future__ import annotations

import asyncio
import logging

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import DetailLevel, RecallResult
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


class Recency:
    """Sort federated results by ``occurred_at`` desc; return first N."""

    requires_grounding = False

    def __init__(
        self,
        *,
        n: int = 10,
        detail: DetailLevel = DetailLevel.PREVIEW,
        per_source_limit: int = 50,
    ) -> None:
        self._n = n
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

        per_source = await asyncio.gather(*(_safe(s) for s in sources))
        flat = [se for batch in per_source for se in batch]
        # Dedup by episode id (first encounter wins for ordering stability).
        seen, dedup = set(), []
        for se in flat:
            if se.episode.id in seen:
                continue
            seen.add(se.episode.id)
            dedup.append(se)
        dedup.sort(key=lambda se: se.episode.occurred_at, reverse=True)
        return RecallResult(episodes=dedup[: self._n])


__all__ = ["Recency"]

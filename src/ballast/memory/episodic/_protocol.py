"""``EpisodicSource`` Protocol — one source of episodic facts.

Apps register many sources in ``EpisodicMemory``. The facade fans out
recall in parallel; a ``RecallStrategy`` merges + reduces.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, Episode, ScoredEpisode,
)


@runtime_checkable
class EpisodicSource(Protocol):
    """Owns ``recall`` / ``hydrate`` / ``remember`` for one backing."""

    name: str

    async def recall(
        self, *,
        intent: str,
        scope: Scope,
        k: int,
        detail: DetailLevel,
    ) -> list[ScoredEpisode]: ...

    async def hydrate(
        self, episode: Episode, *, detail: DetailLevel,
    ) -> Episode: ...

    async def remember(self, episode: Episode) -> None: ...


__all__ = ["EpisodicSource"]

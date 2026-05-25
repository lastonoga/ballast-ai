"""Episode wire-contract: types every source / strategy / consumer speaks."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ballast.grounded import Ref
from ballast.memory._scope import Scope


class DetailLevel(StrEnum):
    """Hydration level. Comparable: ``FULL >= SUMMARY >= PREVIEW``."""

    PREVIEW = "preview"   # 1-2 lines, cheap, always present
    SUMMARY = "summary"   # paragraph-level
    FULL    = "full"      # complete trajectory (messages + tool calls + outputs)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, DetailLevel):
            return NotImplemented
        order = (DetailLevel.PREVIEW, DetailLevel.SUMMARY, DetailLevel.FULL)
        return order.index(self) >= order.index(other)


class Episode(BaseModel):
    """One unit of recallable past activity, source-agnostic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    source: str
    occurred_at: datetime
    scope: Scope

    preview: str
    summary: str | None = None
    full:    dict[str, Any] | None = None

    references: list[Ref[Any]] = []
    metadata: dict[str, Any] = {}


class ScoredEpisode(BaseModel):
    """Episode + source-relative score (a merger normalizes across sources)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode: Episode
    score:   float


class RecallResult(BaseModel):
    """Output of ``EpisodicMemory.episodic_for(...)``.

    ``.references`` aggregates all Ref[T] across all episodes — ready-to-go
    ground-set for ``GroundedAgent`` when this is passed as ``context=[…]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episodes: list[ScoredEpisode] = []

    @property
    def references(self) -> list[Ref[Any]]:
        return [r for se in self.episodes for r in se.episode.references]


__all__ = [
    "DetailLevel", "Episode", "RecallResult", "ScoredEpisode",
]

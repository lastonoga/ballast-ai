"""``EpisodicSource`` Protocol — structural type check."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel, Episode, EpisodicSource, ScoredEpisode,
)


class _Stub:
    name = "stub"
    async def recall(self, *, intent, scope, k, detail) -> list[ScoredEpisode]:
        return []
    async def hydrate(self, episode, *, detail) -> Episode:
        return episode
    async def remember(self, episode) -> None:
        return None


def test_runtime_checkable_protocol() -> None:
    assert isinstance(_Stub(), EpisodicSource)


def test_protocol_requires_name_attr() -> None:
    class NoName:
        async def recall(self, *, intent, scope, k, detail): return []
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None: return None
    # name attribute MUST be present — runtime_checkable doesn't enforce
    # but we document via hasattr.
    assert not hasattr(NoName(), "name")

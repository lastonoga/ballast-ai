"""``RecallStrategy`` Protocol — structural type for strategy impls."""
from __future__ import annotations

from ballast.memory import Scope
from ballast.memory.episodic import EpisodicSource, RecallResult
from ballast.memory.episodic.strategies import RecallStrategy


class _Stub:
    requires_grounding = False
    async def execute(self, *, intent, sources, scope) -> RecallResult:
        return RecallResult(episodes=[])


def test_runtime_checkable() -> None:
    assert isinstance(_Stub(), RecallStrategy)

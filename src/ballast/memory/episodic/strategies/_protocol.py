"""``RecallStrategy`` — pluggable strategy for recall reduction."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import RecallResult
from ballast.memory.episodic._protocol import EpisodicSource


@runtime_checkable
class RecallStrategy(Protocol):
    """Federates per-source recall into one RecallResult.

    Set ``requires_grounding = True`` if the strategy expects every
    Episode to carry ``references`` (so an empty ``references`` set
    surfaces as a warning rather than silent grounding collapse).
    """

    requires_grounding: bool

    async def execute(
        self, *,
        intent: str,
        sources: list[EpisodicSource],
        scope: Scope,
    ) -> RecallResult: ...


__all__ = ["RecallStrategy"]

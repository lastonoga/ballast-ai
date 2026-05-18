from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.capabilities.helpers.semantic_deduper import SemanticDeduper

OutT = TypeVar("OutT")


class TypedLoopGuard(Generic[OutT]):
    """Loop detector for typed Pattern outputs (SP5).

    Pattern code calls `.check(output)` between iterations. Each selector
    field is checked through its OWN deduper instance, so a list of fields
    detects loops independently per field. Loop in ANY field raises.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        selector: Callable[[OutT], str | list[str]],
        threshold: float = 0.95,
        window: int = 3,
    ) -> None:
        self.embedder = embedder
        self.selector = selector
        self.threshold = threshold
        self.window = window
        # Per-index deduper so list selectors get independent loop detection per field.
        self._dedupers: dict[int, SemanticDeduper] = {}

    def _deduper_for(self, index: int) -> SemanticDeduper:
        if index not in self._dedupers:
            self._dedupers[index] = SemanticDeduper(self.embedder)
        return self._dedupers[index]

    async def check(self, output: OutT) -> None:
        snapshots = self.selector(output)
        if isinstance(snapshots, str):
            snapshots = [snapshots]
        for i, snap in enumerate(snapshots):
            await self._deduper_for(i).add_and_check(
                snap, threshold=self.threshold, window=self.window
            )

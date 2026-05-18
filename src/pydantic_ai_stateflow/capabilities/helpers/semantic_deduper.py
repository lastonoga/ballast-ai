from __future__ import annotations

import math
from collections import deque

from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder


class SemanticLoopDetected(Exception):  # noqa: N818
    """Raised when a sliding window of snapshots is too similar (loop / repeat)."""

    def __init__(self, snapshot: str, similarities: list[float] | None = None) -> None:
        self.snapshot = snapshot
        self.similarities = similarities or []
        super().__init__(
            f"SemanticLoopDetected: snapshot={snapshot!r} similarities={self.similarities}"
        )


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns 0 if either is zero."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticDeduper:
    """Sliding-window cosine-similarity loop detector.

    Used by SemanticLoopDetector (L1, raw model response) and by
    TypedLoopGuard (L2, typed output between Pattern iterations).
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._history: deque[list[float]] = deque()

    async def add_and_check(self, snapshot: str, *, threshold: float, window: int) -> None:
        """Embed `snapshot`, slide window, raise SemanticLoopDetected on match."""
        emb = await self._embedder.embed(snapshot)
        # Slide window FIRST so we never compare beyond `window` entries
        while len(self._history) >= window:
            self._history.popleft()
        # Detection: if appending this embedding fills the window AND
        # all existing entries are sufficiently similar to this one, fire.
        if len(self._history) >= window - 1 and self._history:
            sims = [_cosine(emb, prev) for prev in self._history]
            if all(s >= threshold for s in sims):
                self._history.append(emb)
                raise SemanticLoopDetected(snapshot=snapshot[:200], similarities=sims)
        self._history.append(emb)

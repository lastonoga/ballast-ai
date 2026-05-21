from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.patterns.semantic_dedup.config import SemanticDedupConfig

ItemT = TypeVar("ItemT")

Projector = Callable[[ItemT], str]
"""Application supplies how to turn ``ItemT`` into the string we embed.

A lambda is the canonical form — ``lambda i: f"{i.title}\\n{i.body}"``
— because it lets callers slice / normalize / join nested fields
without inflating ``SemanticDedupConfig`` with a mini DSL.
"""


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 when either vector has zero norm.

    Lives in this module rather than in
    ``capabilities.helpers.semantic_deduper`` to avoid a cycle: the
    pattern depends on ``Embedder`` but should not pull in the unrelated
    loop-detector class. The duplication is one short function.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticDedup(Generic[ItemT]):
    """Greedy near-duplicate filter over a list of items.

    Embeds the result of ``projector(item)`` for every item, then walks
    left-to-right collapsing any item whose embedding is within
    ``config.threshold`` cosine of a previously-kept item. Output
    preserves input order.

    Satisfies the structural ``Pattern[list[ItemT], list[ItemT]]``
    contract.

    Examples:
        >>> dedup = SemanticDedup[TodoIdea](
        ...     embedder=my_embedder,
        ...     projector=lambda i: f"{i.title}\\n{i.body}",
        ...     config=SemanticDedupConfig(threshold=0.88, keep="longest"),
        ... )
        >>> survivors = await dedup.run(candidates)

    Anti-patterns (intentionally documented here so future readers see
    them before reaching for the wrong knob):

    * **DON'T** confuse this with
      ``capabilities/helpers/SemanticDeduper`` — that's a sliding-
      window LOOP detector that raises on a repeating stream. This is
      a one-shot batch FILTER over a collection.
    * **DON'T** set ``threshold`` below ``0.85``. False positives eat
      the diversity that earlier stages worked to produce; for creative
      / brainstorm pipelines stay in ``[0.88, 0.92]``.
    * **DON'T** rely on this for N > 1000 items in the hot path. The
      O(N²) pairwise comparison wakes up. Switch to an ANN / LSH index
      or pre-cluster cheaply (e.g. by first token bigram) before
      handing the buckets here.
    """

    name: ClassVar[str] = "semantic_dedup"

    def __init__(
        self,
        *,
        embedder: Embedder,
        projector: Projector[ItemT],
        config: SemanticDedupConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._projector = projector
        self._config = config or SemanticDedupConfig()

    @traced(TraceName.PATTERN_SEMANTIC_DEDUP, attrs=lambda self, items: {
        "pattern": self.name,
        "input_count": len(items),
        "threshold": self._config.threshold,
        "keep": self._config.keep,
    })
    async def run(self, items: list[ItemT]) -> list[ItemT]:
        if not items:
            return []
        projections = [self._projector(item) for item in items]
        embeddings = await self._embedder.embed_batch(projections)
        kept_idx: list[int] = []
        for i, emb in enumerate(embeddings):
            duplicate_of: int | None = None
            for j in kept_idx:
                if _cosine(emb, embeddings[j]) >= self._config.threshold:
                    duplicate_of = j
                    break
            if duplicate_of is None:
                kept_idx.append(i)
            elif self._config.keep == "longest" and len(projections[i]) > len(
                projections[duplicate_of],
            ):
                # Swap: drop the previously-kept shorter one, keep this.
                kept_idx[kept_idx.index(duplicate_of)] = i
        return [items[i] for i in kept_idx]

"""``Cluster`` strategy — k-means dedup; one medoid per cluster.

Uses ``Embedder`` (existing framework Protocol) to vectorize episode
previews; minimal k-means without external numpy dep — we only need
correctness for small N, not speed.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random

from ballast.capabilities.helpers.embedder import Embedder
from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, RecallResult, ScoredEpisode,
)
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _kmeans_assign(
    points: list[list[float]], k: int, *, max_iter: int = 20, seed: int = 0,
) -> list[int]:
    """Returns cluster-id per point. Distance = 1 - cosine_sim."""
    if k <= 0: return [0] * len(points)
    rng = random.Random(seed)
    centroids = [points[i] for i in rng.sample(range(len(points)), k=min(k, len(points)))]
    assignments = [0] * len(points)
    for _ in range(max_iter):
        new_assignments = [
            min(range(len(centroids)),
                key=lambda c: 1.0 - _cosine(p, centroids[c]))
            for p in points
        ]
        if new_assignments == assignments: break
        assignments = new_assignments
        for c in range(len(centroids)):
            cluster_pts = [points[i] for i, a in enumerate(assignments) if a == c]
            if not cluster_pts: continue
            dim = len(cluster_pts[0])
            centroids[c] = [
                sum(p[d] for p in cluster_pts) / len(cluster_pts) for d in range(dim)
            ]
    return assignments


class Cluster:
    """Semantic dedup — one episode per cluster (medoid)."""

    requires_grounding = False

    def __init__(
        self,
        *,
        n_clusters: int = 5,
        embedder: Embedder,
        detail: DetailLevel = DetailLevel.SUMMARY,
        per_source_limit: int = 50,
    ) -> None:
        self._n = n_clusters
        self._embedder = embedder
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
        flat: list[ScoredEpisode] = [se for batch in per_source for se in batch]
        # Dedup by id before clustering.
        seen, dedup = set(), []
        for se in flat:
            if se.episode.id in seen: continue
            seen.add(se.episode.id); dedup.append(se)
        if not dedup:
            return RecallResult(episodes=[])
        embs = await self._embedder.embed_batch([
            se.episode.summary or se.episode.preview for se in dedup
        ])
        assigns = _kmeans_assign(embs, self._n)
        # Pick highest-score representative per cluster.
        reps: dict[int, ScoredEpisode] = {}
        for se, cid in zip(dedup, assigns, strict=True):
            if cid not in reps or se.score > reps[cid].score:
                reps[cid] = se
        return RecallResult(
            episodes=sorted(reps.values(), key=lambda se: se.score, reverse=True),
        )


__all__ = ["Cluster"]

"""Cross-source mergers for federated episodic recall.

Default: ``RRFMerger`` (Reciprocal Rank Fusion) — IR-standard, doesn't
require comparable scores across sources.
"""
from __future__ import annotations

from typing import Protocol

from ballast.memory.episodic._models import Episode, ScoredEpisode


class ScoreMerger(Protocol):
    """Merge per-source ScoredEpisode lists into a single ranked list."""

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]: ...


class RRFMerger(ScoreMerger):
    """Reciprocal Rank Fusion: ``score(d) = sum_s 1 / (k + rank_s(d))``.

    Where ``rank_s(d)`` is the rank (1-indexed) of d in source s, or
    infinity if d isn't in source s. ``k=60`` is the canonical RRF
    constant. Dedupes by episode id; episode chosen is the first
    encounter (sources order-independent — we sum contributions).
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]:
        agg: dict[str, tuple[Episode, float]] = {}
        for results in results_by_source.values():
            for rank, scored in enumerate(results, start=1):
                ep_id = scored.episode.id
                contribution = 1.0 / (self._k + rank)
                if ep_id in agg:
                    existing_ep, existing_score = agg[ep_id]
                    agg[ep_id] = (existing_ep, existing_score + contribution)
                else:
                    agg[ep_id] = (scored.episode, contribution)
        return sorted(
            (ScoredEpisode(episode=ep, score=sc) for ep, sc in agg.values()),
            key=lambda se: se.score,
            reverse=True,
        )


class WeightedMerger(ScoreMerger):
    """Per-source weighted scores. Apps hint relative trust."""

    def __init__(
        self, weights: dict[str, float], *, normalize: bool = True,
    ) -> None:
        if normalize:
            total = sum(weights.values()) or 1.0
            weights = {k: v / total for k, v in weights.items()}
        self._weights = weights

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]:
        agg: dict[str, tuple[Episode, float]] = {}
        for src, results in results_by_source.items():
            w = self._weights.get(src, 0.0)
            for scored in results:
                ep_id = scored.episode.id
                contrib = scored.score * w
                if ep_id in agg:
                    existing_ep, existing_score = agg[ep_id]
                    agg[ep_id] = (existing_ep, existing_score + contrib)
                else:
                    agg[ep_id] = (scored.episode, contrib)
        return sorted(
            (ScoredEpisode(episode=ep, score=sc) for ep, sc in agg.values()),
            key=lambda se: se.score, reverse=True,
        )


class RawScoreMerger(ScoreMerger):
    """Simple union + sort by raw score. Requires score comparability
    across sources — use only when all sources produce calibrated scores."""

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]:
        merged: list[ScoredEpisode] = []
        seen: set[str] = set()
        for results in results_by_source.values():
            for scored in results:
                if scored.episode.id not in seen:
                    merged.append(scored)
                    seen.add(scored.episode.id)
        return sorted(merged, key=lambda se: se.score, reverse=True)


__all__ = ["RRFMerger", "RawScoreMerger", "ScoreMerger", "WeightedMerger"]

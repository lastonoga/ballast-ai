"""``RRFMerger`` — Reciprocal Rank Fusion, IR-standard cross-source merge."""
from __future__ import annotations

from datetime import UTC, datetime

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic._mergers import RRFMerger


def _ep(id_: str) -> Episode:
    return Episode(
        id=id_, source="x", occurred_at=datetime.now(UTC),
        scope=Scope(), preview="p",
    )


def test_rrf_merges_two_sources_with_overlap() -> None:
    """Overlapping episode appears once with combined RRF score."""
    a, b, c = _ep("a"), _ep("b"), _ep("c")
    src1 = [ScoredEpisode(episode=a, score=0.9),
            ScoredEpisode(episode=b, score=0.6)]
    src2 = [ScoredEpisode(episode=a, score=0.8),
            ScoredEpisode(episode=c, score=0.4)]

    merged = RRFMerger(k=60).merge({"s1": src1, "s2": src2})
    ids = [se.episode.id for se in merged]
    assert "a" in ids
    assert len(set(ids)) == len(ids)   # dedup'd


def test_rrf_score_higher_for_higher_combined_rank() -> None:
    a = _ep("a"); b = _ep("b")
    src1 = [ScoredEpisode(episode=a, score=0.9),
            ScoredEpisode(episode=b, score=0.5)]
    src2 = [ScoredEpisode(episode=a, score=0.8),
            ScoredEpisode(episode=b, score=0.6)]
    merged = RRFMerger().merge({"s1": src1, "s2": src2})
    by_id = {se.episode.id: se.score for se in merged}
    assert by_id["a"] > by_id["b"]   # both rank-1 in each source > both rank-2


def test_rrf_empty_inputs() -> None:
    assert RRFMerger().merge({}) == []
    assert RRFMerger().merge({"s1": []}) == []

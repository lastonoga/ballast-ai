"""Episode wire-contract: DetailLevel, Episode, ScoredEpisode, RecallResult."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from ballast.grounded import Ref
from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel, Episode, RecallResult, ScoredEpisode,
)


def _now() -> datetime: return datetime(2026, 5, 25, tzinfo=UTC)

_UUID1 = UUID("00000000-0000-0000-0000-000000000001")
_UUID2 = UUID("00000000-0000-0000-0000-000000000002")
_UUID3 = UUID("00000000-0000-0000-0000-000000000003")


def test_detail_level_string_enum() -> None:
    assert DetailLevel.PREVIEW.value == "preview"
    assert DetailLevel.SUMMARY.value == "summary"
    assert DetailLevel.FULL.value == "full"
    # Ordering for comparison (>=) — implemented via int conversion.
    assert DetailLevel.SUMMARY >= DetailLevel.PREVIEW
    assert DetailLevel.FULL >= DetailLevel.SUMMARY


def test_episode_minimal() -> None:
    ep = Episode(
        id="thread:abc:turn:0", source="thread",
        occurred_at=_now(), scope=Scope(user_id="u-1"),
        preview="user asked about ML",
    )
    assert ep.preview == "user asked about ML"
    assert ep.summary is None
    assert ep.full is None
    assert ep.references == []


def test_episode_with_references() -> None:
    note_ref = Ref[str](_UUID1)
    ep = Episode(
        id="ep-1", source="vector", occurred_at=_now(),
        scope=Scope(user_id="u-1"), preview="...",
        references=[note_ref],
    )
    assert len(ep.references) == 1


def test_recall_result_references_aggregates() -> None:
    note1 = Ref[str](_UUID1)
    note2 = Ref[str](_UUID2)
    note3 = Ref[str](_UUID3)
    ep_a = Episode(
        id="a", source="x", occurred_at=_now(),
        scope=Scope(), preview="p", references=[note1, note2],
    )
    ep_b = Episode(
        id="b", source="x", occurred_at=_now(),
        scope=Scope(), preview="p", references=[note3],
    )
    rr = RecallResult(episodes=[
        ScoredEpisode(episode=ep_a, score=0.9),
        ScoredEpisode(episode=ep_b, score=0.5),
    ])
    refs = rr.references
    assert len(refs) == 3

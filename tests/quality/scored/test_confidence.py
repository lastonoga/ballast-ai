"""Confidence label + helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ballast.quality.scored._confidence import (
    Confidence,
    aggregate_by_confidence,
    filter_by_min_confidence,
    label_rank,
    label_to_float,
    rank_by_confidence,
)


@dataclass
class _Scored:
    """Minimal stand-in for Scored[T] — helpers duck-type on .confidence/.value."""
    value: str
    confidence: Confidence


def test_label_to_float_mapping() -> None:
    assert label_to_float("low") == pytest.approx(0.33)
    assert label_to_float("medium") == pytest.approx(0.66)
    assert label_to_float("high") == pytest.approx(1.0)


def test_label_rank_ordering() -> None:
    assert label_rank("low") == 0
    assert label_rank("medium") == 1
    assert label_rank("high") == 2


def test_aggregate_buckets_items_by_label() -> None:
    items = [
        _Scored("a", "high"),
        _Scored("b", "low"),
        _Scored("c", "high"),
        _Scored("d", "medium"),
    ]
    out = aggregate_by_confidence(items)
    assert out == {
        "low": ["b"],
        "medium": ["d"],
        "high": ["a", "c"],
    }


def test_aggregate_empty_returns_empty_buckets() -> None:
    out = aggregate_by_confidence([])
    assert out == {"low": [], "medium": [], "high": []}


def test_filter_by_min_confidence_includes_threshold() -> None:
    items = [
        _Scored("a", "low"),
        _Scored("b", "medium"),
        _Scored("c", "high"),
    ]
    assert [it.value for it in filter_by_min_confidence(items, "medium")] == ["b", "c"]
    assert [it.value for it in filter_by_min_confidence(items, "high")] == ["c"]
    assert [it.value for it in filter_by_min_confidence(items, "low")] == ["a", "b", "c"]


def test_rank_by_confidence_descending_with_stable_sort() -> None:
    items = [
        _Scored("a", "medium"),
        _Scored("b", "high"),
        _Scored("c", "medium"),
        _Scored("d", "low"),
        _Scored("e", "high"),
    ]
    ranked = rank_by_confidence(items)
    assert [it.value for it in ranked] == ["b", "e", "a", "c", "d"]


def test_rank_by_confidence_with_secondary_key() -> None:
    items = [
        _Scored("z", "high"),
        _Scored("a", "high"),
        _Scored("m", "low"),
    ]
    ranked = rank_by_confidence(items, secondary_key=lambda it: it.value)
    assert [it.value for it in ranked] == ["a", "z", "m"]


def test_label_rank_raises_on_unknown_label() -> None:
    with pytest.raises(KeyError):
        label_rank("bogus")  # type: ignore[arg-type]

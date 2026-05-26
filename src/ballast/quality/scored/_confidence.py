"""Default ``Confidence`` Literal + helpers.

Helpers are hardcoded for the default 3-bin Literal labels. Apps using
``Scored[T, CustomConfidenceT]`` write their own helpers — trivial.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from ballast.quality.scored._model import Scored


Confidence = Literal["low", "medium", "high"]
"""Default 3-bin labeled confidence. Pluggable via ``Scored[T, ConfidenceT]``."""

_LABEL_TO_FLOAT: dict[Confidence, float] = {"low": 0.33, "medium": 0.66, "high": 1.0}
_LABEL_ORDER:    dict[Confidence, int]   = {"low": 0, "medium": 1, "high": 2}


def label_to_float(label: Confidence) -> float:
    """Map a default Confidence label into a [0, 1] float for metrics."""
    return _LABEL_TO_FLOAT[label]


def label_rank(label: Confidence) -> int:
    """Map a default Confidence label into an ordinal rank (low=0, high=2)."""
    return _LABEL_ORDER[label]


T = TypeVar("T")


def aggregate_by_confidence(items: list["Scored[T]"]) -> dict[Confidence, list[T]]:
    """Bucket items by their confidence label. Returns a dict with all
    three keys always present (empty lists if no entries)."""
    out: dict[Confidence, list[T]] = {"low": [], "medium": [], "high": []}
    for it in items:
        out[it.confidence].append(it.value)
    return out


def filter_by_min_confidence(
    items: list["Scored[T]"], min_label: Confidence,
) -> list["Scored[T]"]:
    """Keep only items with confidence rank >= ``min_label`` rank."""
    threshold = label_rank(min_label)
    return [it for it in items if label_rank(it.confidence) >= threshold]


def rank_by_confidence(
    items: list["Scored[T]"], *,
    secondary_key: Callable[["Scored[T]"], Any] | None = None,
) -> list["Scored[T]"]:
    """Sort descending by confidence (high → low). Stable sort; optional
    secondary key for tie-breaking."""
    def _key(it: "Scored[T]") -> tuple[int, Any]:
        secondary = secondary_key(it) if secondary_key else 0
        return (-label_rank(it.confidence), secondary)

    return sorted(items, key=_key)


__all__ = [
    "Confidence",
    "aggregate_by_confidence",
    "filter_by_min_confidence",
    "label_rank",
    "label_to_float",
    "rank_by_confidence",
]

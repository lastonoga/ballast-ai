"""``DriftVerdictBase`` + ``DefaultDriftVerdict`` — verdict types.

Apps may subclass ``DriftVerdictBase`` to add domain-specific verdict
fields. The framework only reads ``should_interrupt`` and ``reason``;
everything else is for the app's own handlers / observability.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DriftVerdictBase(BaseModel):
    """Minimum contract — framework reads these two fields."""

    should_interrupt: bool
    """If True, framework runs all configured ``DriftHandler``s."""

    reason: str
    """CoT обоснование (для логов / HITL контекста)."""


class DefaultDriftVerdict(DriftVerdictBase):
    """Rich default verdict — used when caller doesn't supply a custom one."""

    score: float
    """0.0 = полный дрейф ... 1.0 = на цели."""

    category: Literal["on_track", "loose", "drifted"]
    """Coarse-grained label for metrics / dashboards."""

    suggested_action: str | None = None
    """Optional next-step recommendation from the judge."""


__all__ = ["DriftVerdictBase", "DefaultDriftVerdict"]

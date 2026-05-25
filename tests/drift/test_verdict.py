"""DriftVerdictBase + DefaultDriftVerdict — verdict type contract."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ballast.drift._verdict import DriftVerdictBase, DefaultDriftVerdict


def test_base_requires_should_interrupt_and_reason() -> None:
    v = DriftVerdictBase(should_interrupt=True, reason="drifted off-topic")
    assert v.should_interrupt is True
    assert v.reason == "drifted off-topic"


def test_base_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        DriftVerdictBase(should_interrupt=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        DriftVerdictBase(reason="x")  # type: ignore[call-arg]


def test_default_verdict_adds_score_category_action() -> None:
    v = DefaultDriftVerdict(
        should_interrupt=False, reason="on track",
        score=0.9, category="on_track",
    )
    assert v.score == 0.9
    assert v.category == "on_track"
    assert v.suggested_action is None


def test_default_verdict_category_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        DefaultDriftVerdict(
            should_interrupt=False, reason="x",
            score=0.5, category="bogus",  # type: ignore[arg-type]
        )


def test_default_verdict_is_subclass_of_base() -> None:
    assert issubclass(DefaultDriftVerdict, DriftVerdictBase)
    v = DefaultDriftVerdict(
        should_interrupt=True, reason="r",
        score=0.0, category="drifted",
    )
    assert isinstance(v, DriftVerdictBase)

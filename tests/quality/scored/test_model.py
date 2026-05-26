"""Scored[T, ConfidenceT] generic BaseModel."""
from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from ballast.quality.scored._model import Scored


class _Note(BaseModel):
    title: str
    body: str


def test_scored_basic_instantiation() -> None:
    note = _Note(title="t", body="b")
    s = Scored[_Note](value=note, rationale="from doc", confidence="high")
    assert s.value is note
    assert s.rationale == "from doc"
    assert s.confidence == "high"


def test_scored_with_list_value() -> None:
    notes = [_Note(title="t1", body="b1"), _Note(title="t2", body="b2")]
    s = Scored[list[_Note]](value=notes, rationale="batched", confidence="medium")
    assert len(s.value) == 2
    assert s.confidence == "medium"


def test_scored_frozen_assignment_raises() -> None:
    s = Scored[str](value="x", rationale="r", confidence="low")
    with pytest.raises(ValidationError):
        s.confidence = "high"  # type: ignore[misc]


def test_scored_rationale_required() -> None:
    with pytest.raises(ValidationError):
        Scored[str](value="x", confidence="high")  # type: ignore[call-arg]


def test_scored_rejects_invalid_confidence_label() -> None:
    with pytest.raises(ValidationError):
        Scored[str](value="x", rationale="r", confidence="bogus")  # type: ignore[arg-type]


def test_scored_empty_rationale_allowed() -> None:
    s = Scored[str](value="x", rationale="", confidence="low")
    assert s.rationale == ""


def test_scored_with_custom_confidence_int() -> None:
    s = Scored[str, int](value="x", rationale="r", confidence=4)
    assert s.confidence == 4


def test_scored_with_custom_confidence_literal() -> None:
    Binary = Literal["safe", "uncertain"]
    s = Scored[str, Binary](value="x", rationale="r", confidence="safe")
    assert s.confidence == "safe"


def test_scored_json_schema_has_required_fields() -> None:
    schema = Scored[_Note].model_json_schema()
    required = set(schema.get("required", []))
    assert "value" in required
    assert "rationale" in required
    assert "confidence" in required


def test_scored_dump_roundtrip() -> None:
    s = Scored[str](value="x", rationale="r", confidence="medium")
    dumped = s.model_dump()
    assert dumped == {"value": "x", "rationale": "r", "confidence": "medium"}
    s2 = Scored[str].model_validate(dumped)
    assert s2 == s

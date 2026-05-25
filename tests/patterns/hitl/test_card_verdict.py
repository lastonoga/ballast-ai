"""``CardVerdict[OutT]`` + ``__hitl_kind__`` registry + signals."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    card_kind_registry,
    register_card_kind,
)


class _Note(BaseModel):
    __hitl_kind__ = "note.create"
    title: str
    body: str


def test_verdict_typed_modified_field() -> None:
    v = CardVerdict[_Note](
        decision="approve",
        modified=_Note(title="t", body="b"),
        answered_at=datetime(2026, 5, 25, tzinfo=UTC),
    )
    assert v.modified is not None and v.modified.title == "t"


def test_verdict_reject_no_modified() -> None:
    v = CardVerdict[_Note](
        decision="reject",
        answered_at=datetime(2026, 5, 25, tzinfo=UTC),
    )
    assert v.modified is None
    assert v.decision == "reject"


def test_register_card_kind_indexes_by_hitl_kind_attr() -> None:
    register_card_kind(_Note)
    assert card_kind_registry["note.create"] is _Note


def test_register_card_kind_requires_attr() -> None:
    class NoKind(BaseModel):
        x: str
    with pytest.raises(AttributeError, match="__hitl_kind__"):
        register_card_kind(NoKind)

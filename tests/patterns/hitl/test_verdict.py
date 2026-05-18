from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.patterns.hitl.verdict import HelperVerdict


class _Ctx(BaseModel):
    note: str


# Module-level alias REQUIRED (project memory: parameterized generics that
# cross @DBOS.workflow() boundaries need a module-level alias).
_CtxVerdict = HelperVerdict[_Ctx]
_NoneVerdict = HelperVerdict[None]


def test_basic_construction_with_typed_context() -> None:
    v = _CtxVerdict(
        rationale="lgtm",
        confidence=0.9,
        conversation_turn_count=3,
        tools_invoked=["cite", "approve"],
        context=_Ctx(note="hi"),
    )
    assert v.rationale == "lgtm"
    assert v.confidence == 0.9
    assert v.conversation_turn_count == 3
    assert v.tools_invoked == ["cite", "approve"]
    assert v.autopilot_eligible is False
    assert v.autopilot_confidence is None
    assert v.context is not None
    assert v.context.note == "hi"


def test_autopilot_fields_optional() -> None:
    v = _CtxVerdict(
        rationale="r", confidence=1.0,
        conversation_turn_count=0, tools_invoked=[],
        autopilot_eligible=True, autopilot_confidence=0.42,
    )
    assert v.autopilot_eligible is True
    assert v.autopilot_confidence == 0.42


def test_none_context_form() -> None:
    v = _NoneVerdict(
        rationale="ok", confidence=1.0,
        conversation_turn_count=1, tools_invoked=[],
    )
    assert v.context is None


def test_rejects_wrong_context_type() -> None:
    class _Other(BaseModel):
        pass
    with pytest.raises(ValidationError):
        _CtxVerdict(
            rationale="r", confidence=1.0,
            conversation_turn_count=0, tools_invoked=[],
            context=_Other(),  # type: ignore[arg-type]
        )


def test_round_trip_via_model_dump_and_validate() -> None:
    v = _CtxVerdict(
        rationale="r", confidence=0.5,
        conversation_turn_count=2, tools_invoked=["x"],
        context=_Ctx(note="n"),
    )
    dumped = v.model_dump(mode="json")
    restored = _CtxVerdict.model_validate(dumped)
    assert restored == v

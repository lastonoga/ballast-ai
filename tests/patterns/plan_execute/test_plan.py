"""Plan + PlannedStep + DAG validator."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ballast.patterns.plan_execute._plan import Plan, PlannedStep


def _step(id: str, deps: list[str] = ()) -> PlannedStep:
    return PlannedStep(id=id, kind="llm", params={}, depends_on=list(deps))


def test_empty_plan_is_valid() -> None:
    p = Plan(steps=[])
    assert p.steps == []
    assert p.rationale == ""


def test_linear_chain_is_valid() -> None:
    p = Plan(steps=[_step("a"), _step("b", ["a"]), _step("c", ["b"])])
    assert len(p.steps) == 3


def test_diamond_dag_is_valid() -> None:
    p = Plan(steps=[
        _step("a"), _step("b", ["a"]), _step("c", ["a"]), _step("d", ["b", "c"]),
    ])
    assert len(p.steps) == 4


def test_duplicate_step_id_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Plan(steps=[_step("a"), _step("a")])


def test_dangling_dep_rejected() -> None:
    with pytest.raises(ValidationError, match="dangling"):
        Plan(steps=[_step("a", ["nonexistent"])])


def test_cycle_detected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Plan(steps=[_step("a", ["b"]), _step("b", ["a"])])


def test_self_loop_detected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Plan(steps=[_step("a", ["a"])])


def test_rationale_field_optional() -> None:
    p = Plan(steps=[], rationale="initial plan")
    assert p.rationale == "initial plan"


def test_planned_step_required_fields() -> None:
    s = PlannedStep(id="a", kind="llm", params={"x": 1})
    assert s.id == "a"
    assert s.depends_on == []
    assert s.description == ""

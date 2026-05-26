"""StepRegistry — apps populate, framework dispatches."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._registry import StepRegistry


class _FakeStep:
    async def execute(self, plan_input, dep_outputs, ctx): return "out"


def test_register_get_step() -> None:
    r = StepRegistry()
    s = _FakeStep()
    r.register_step("custom", s)
    assert r.get_step("custom") is s


def test_get_unknown_step_raises_helpful_keyerror() -> None:
    r = StepRegistry()
    r.register_step("foo", _FakeStep())
    with pytest.raises(KeyError, match="bar") as exc:
        r.get_step("bar")
    assert "foo" in str(exc.value)
    assert "available" in str(exc.value)


def test_register_get_agent_callable_unit_workflow() -> None:
    r = StepRegistry()
    obj_a, obj_b, obj_c, obj_d = object(), object(), object(), object()
    r.register_agent("ag", obj_a)
    r.register_callable("cb", obj_b)
    r.register_unit("un", obj_c)
    r.register_workflow("wf", obj_d)
    assert r.get_agent("ag") is obj_a
    assert r.get_callable("cb") is obj_b
    assert r.get_unit("un") is obj_c
    assert r.get_workflow("wf") is obj_d


def test_get_unknown_agent_callable_unit_workflow_raises_keyerror() -> None:
    r = StepRegistry()
    r.register_agent("foo", object())
    with pytest.raises(KeyError, match="bar"):
        r.get_agent("bar")
    with pytest.raises(KeyError):
        r.get_callable("nope")
    with pytest.raises(KeyError):
        r.get_unit("nope")
    with pytest.raises(KeyError):
        r.get_workflow("nope")


def test_with_defaults_preregisters_four_step_kinds() -> None:
    r = StepRegistry.with_defaults()
    assert r.get_step("llm") is not None
    assert r.get_step("callable") is not None
    assert r.get_step("unit") is not None
    assert r.get_step("workflow") is not None

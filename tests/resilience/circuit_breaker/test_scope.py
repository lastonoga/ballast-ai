"""Built-in ScopeKey helpers."""
from __future__ import annotations

from ballast.resilience.circuit_breaker._scope import (
    global_scope, per_step_scope, per_tool_scope,
)


def test_global_scope_constant() -> None:
    assert global_scope({}) == "global"
    assert global_scope({"tool_name": "x"}) == "global"


def test_per_tool_scope_uses_tool_name() -> None:
    assert per_tool_scope({"tool_name": "search"}) == "tool:search"
    assert per_tool_scope({}) == "tool:unknown"


def test_per_step_scope_uses_step_id() -> None:
    assert per_step_scope({"step_id": "s1"}) == "step:s1"
    assert per_step_scope({}) == "step:unknown"

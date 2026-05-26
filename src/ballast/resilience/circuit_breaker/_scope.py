"""Built-in ``ScopeKey`` helpers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def global_scope(_ctx: Mapping[str, Any]) -> str:
    """All invocations share one scope. The simplest case."""
    return "global"


def per_tool_scope(ctx: Mapping[str, Any]) -> str:
    """One scope per tool name. Wire when CB protects tool calls."""
    return f"tool:{ctx.get('tool_name', 'unknown')}"


def per_step_scope(ctx: Mapping[str, Any]) -> str:
    """One scope per PlanAndExecute step id. Wire when CB protects DAG nodes."""
    return f"step:{ctx.get('step_id', 'unknown')}"


__all__ = ["global_scope", "per_step_scope", "per_tool_scope"]

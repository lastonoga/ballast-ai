"""``Plan`` + ``PlannedStep`` — DAG of planner-emitted execution nodes.

``Plan.__init__`` validates the DAG: unique step ids, no dangling
dependencies, no cycles. Apps construct plans either via a typed
planner agent (``Agent[None, Plan]``) or manually for testing.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class PlannedStep(BaseModel):
    """One node in the DAG. Planner emits these; executor consumes."""

    id: str
    """Unique within plan; planner picks."""

    kind: str
    """Registry key — ``"llm"`` / ``"callable"`` / ``"unit"`` / ``"workflow"`` / custom."""

    params: dict[str, Any] = {}
    """Kind-specific config — e.g. ``{"agent_name": "...", "prompt_template": "..."}``."""

    depends_on: list[str] = []
    """Other ``PlannedStep.id`` values this step depends on. Empty = root."""

    description: str = ""
    """Human-readable rationale from planner — surfaces in logs / observability."""


class Plan(BaseModel):
    """Full DAG. Validated at construction."""

    steps: list[PlannedStep] = []
    rationale: str = ""

    @model_validator(mode="after")
    def _validate_dag(self) -> "Plan":
        ids = [s.id for s in self.steps]
        seen: set[str] = set()
        for sid in ids:
            if sid in seen:
                raise ValueError(f"Plan has duplicate step id: {sid!r}")
            seen.add(sid)

        # Dangling dep check
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in seen:
                    raise ValueError(
                        f"Step {s.id!r} has dangling dependency: "
                        f"{dep!r} (not in plan)"
                    )

        # Cycle detection via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in seen}
        adj: dict[str, list[str]] = {s.id: list(s.depends_on) for s in self.steps}

        def _dfs(node: str) -> None:
            color[node] = GRAY
            for dep in adj[node]:
                if color[dep] == GRAY:
                    raise ValueError(
                        f"Plan has cycle involving step {node!r} → {dep!r}"
                    )
                if color[dep] == WHITE:
                    _dfs(dep)
            color[node] = BLACK

        for sid in seen:
            if color[sid] == WHITE:
                _dfs(sid)

        return self


__all__ = ["Plan", "PlannedStep"]

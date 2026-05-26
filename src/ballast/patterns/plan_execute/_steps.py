"""Built-in ``Step`` implementations: ``LLMStep``, ``CallableStep``,
``UnitStep``, ``WorkflowStep``.

This module ships with stubs in T5; each real impl is added in T6-T9.
"""
from __future__ import annotations

from typing import Any

from ballast.patterns.plan_execute._registry import StepRegistry


class LLMStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("LLMStep — implemented in Task 6")


class CallableStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("CallableStep — implemented in Task 7")


class UnitStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("UnitStep — implemented in Task 8")


class WorkflowStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("WorkflowStep — implemented in Task 9")


__all__ = ["CallableStep", "LLMStep", "UnitStep", "WorkflowStep"]

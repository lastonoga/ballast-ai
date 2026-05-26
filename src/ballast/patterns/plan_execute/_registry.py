"""``StepRegistry`` — apps register agents / callables / units / workflows
under names; planner references them in ``PlannedStep.params``; framework
dispatches via this registry.

``with_defaults()`` pre-registers the four built-in step kinds (``llm``,
``callable``, ``unit``, ``workflow``) so apps only need to register their
own agents / callables / units / workflows by name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ballast.patterns.plan_execute._protocols import Step

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import Agent

    from ballast.coala import CoALAUnit


def _err(kind: str, key: str, available: list[str]) -> KeyError:
    return KeyError(
        f"{kind} {key!r} not registered; available: {sorted(available)}"
    )


class StepRegistry:
    """Apps populate this before ``PlanAndExecute.run()``.

    Registration is name-keyed; planner emits ``step.kind`` + ``step.params``
    referencing those names. Framework dispatches via this registry without
    any reflection or magic.
    """

    def __init__(self) -> None:
        self._steps:     dict[str, Step]                = {}
        self._agents:    dict[str, "Agent[Any, Any]"]   = {}
        self._callables: dict[str, "Callable[..., Any]"] = {}
        self._units:     dict[str, "CoALAUnit"]         = {}
        self._workflows: dict[str, "Callable[..., Any]"] = {}

    # ---- register --------------------------------------------------------

    def register_step(self, kind: str, impl: Step) -> None:
        self._steps[kind] = impl

    def register_agent(self, name: str, agent: "Agent[Any, Any]") -> None:
        self._agents[name] = agent

    def register_callable(self, name: str, fn: "Callable[..., Any]") -> None:
        self._callables[name] = fn

    def register_unit(self, name: str, unit: "CoALAUnit") -> None:
        self._units[name] = unit

    def register_workflow(self, name: str, wf: "Callable[..., Any]") -> None:
        self._workflows[name] = wf

    # ---- get -------------------------------------------------------------

    def get_step(self, kind: str) -> Step:
        if kind not in self._steps:
            raise _err("step kind", kind, list(self._steps))
        return self._steps[kind]

    def get_agent(self, name: str) -> "Agent[Any, Any]":
        if name not in self._agents:
            raise _err("agent", name, list(self._agents))
        return self._agents[name]

    def get_callable(self, name: str) -> "Callable[..., Any]":
        if name not in self._callables:
            raise _err("callable", name, list(self._callables))
        return self._callables[name]

    def get_unit(self, name: str) -> "CoALAUnit":
        if name not in self._units:
            raise _err("unit", name, list(self._units))
        return self._units[name]

    def get_workflow(self, name: str) -> "Callable[..., Any]":
        if name not in self._workflows:
            raise _err("workflow", name, list(self._workflows))
        return self._workflows[name]

    # ---- factory ---------------------------------------------------------

    @classmethod
    def with_defaults(cls) -> "StepRegistry":
        """Pre-register the four built-in step kinds with this registry."""
        from ballast.patterns.plan_execute._steps import (
            CallableStep, LLMStep, UnitStep, WorkflowStep,
        )
        r = cls()
        r.register_step("llm",      LLMStep(r))
        r.register_step("callable", CallableStep(r))
        r.register_step("unit",     UnitStep(r))
        r.register_step("workflow", WorkflowStep(r))
        return r


__all__ = ["StepRegistry"]

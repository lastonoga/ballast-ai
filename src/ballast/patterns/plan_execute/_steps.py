"""Built-in ``Step`` implementations: ``LLMStep``, ``CallableStep``,
``UnitStep``, ``WorkflowStep``.

Each dispatches via ``StepRegistry`` to the actual agent / function /
unit / workflow the app registered under a name. Planner emits
``PlannedStep(kind=..., params={...})``; framework wires it together.
"""
from __future__ import annotations

import re
from typing import Any

from ballast.patterns.plan_execute._registry import StepRegistry


_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(\.[a-zA-Z_][a-zA-Z0-9_]*)?\}")


def _render_prompt(
    template: str, plan_input: Any, dep_outputs: dict[str, Any],
) -> str:
    """f-string-like substitution. Supports:
      {plan_input}            — whole plan_input stringified
      {plan_input.field}      — attribute or dict-key access on plan_input
      {dep_id}                — whole dep output stringified
      {dep_id.field}          — attribute or dict-key access on dep output
    """
    def _resolve(name: str, attr: str | None) -> str:
        if name == "plan_input":
            value = plan_input
        elif name in dep_outputs:
            value = dep_outputs[name]
        else:
            return f"{{{name}{attr or ''}}}"  # leave unresolved literal
        if attr is None:
            return str(value)
        field = attr[1:]  # drop leading '.'
        if hasattr(value, field):
            return str(getattr(value, field))
        if isinstance(value, dict):
            return str(value.get(field, f"{{{name}{attr}}}"))
        return f"{{{name}{attr}}}"

    return _PLACEHOLDER.sub(
        lambda m: _resolve(m.group(1), m.group(2)),
        template,
    )


class LLMStep:
    """Run a registered pydantic-ai Agent with a templated prompt.

    Planner emits:
        PlannedStep(kind="llm", params={
            "agent_name": "<name>",
            "prompt_template": "<text with {plan_input.x} / {dep_id.field}>",
            "output_field": "<optional field name>",
        })
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        agent = self._registry.get_agent(params["agent_name"])
        prompt = _render_prompt(
            params["prompt_template"], plan_input, dep_outputs,
        )
        result = await agent.run(prompt)
        output = result.output
        if "output_field" in params:
            field = params["output_field"]
            if hasattr(output, field):
                output = getattr(output, field)
            elif isinstance(output, dict) and field in output:
                output = output[field]
        return output


class CallableStep:
    """Run a registered async function.

    Planner emits:
        PlannedStep(kind="callable", params={
            "fn_name": "<name>",
            "args": {"k": v, ...},  # optional extra kwargs
        })

    The function is invoked as ``fn(plan_input=..., dep_outputs=..., **args)``.
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        fn = self._registry.get_callable(params["fn_name"])
        extra = params.get("args", {})
        return await fn(
            plan_input=plan_input, dep_outputs=dep_outputs, **extra,
        )


class UnitStep:
    """Run a registered ``CoALAUnit`` through its 4-phase lifecycle.

    Planner emits:
        PlannedStep(kind="unit", params={
            "unit_name": "<name>",
            "input_from": "<dep_id>",  # optional — use dep output instead of plan_input
        })
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        unit = self._registry.get_unit(params["unit_name"])
        unit_input = (
            dep_outputs[params["input_from"]]
            if "input_from" in params
            else plan_input
        )
        observation = await unit.observe(unit_input)
        retrieved = await unit.retrieve(observation)
        out = await unit.act(observation, retrieved)
        await unit.learn(observation, retrieved, out)
        return out


class WorkflowStep:
    """Run a registered async workflow callable as a sub-step.

    The workflow callable is invoked with a single positional argument:
    either the original ``plan_input`` or the named dep output if
    ``input_from`` is set.

    Planner emits:
        PlannedStep(kind="workflow", params={
            "workflow_name": "<name>",
            "input_from": "<dep_id>",  # optional
        })
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        workflow = self._registry.get_workflow(params["workflow_name"])
        wf_input = (
            dep_outputs[params["input_from"]]
            if "input_from" in params
            else plan_input
        )
        return await workflow(wf_input)


__all__ = ["CallableStep", "LLMStep", "UnitStep", "WorkflowStep"]

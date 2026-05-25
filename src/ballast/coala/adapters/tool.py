"""``as_tool`` — adapt CoALAUnit to pydantic-ai Tool."""
from __future__ import annotations

import inspect
from typing import Any

from pydantic_ai import Tool

from ballast.coala._protocol import CoALAUnit


def as_tool(
    unit: CoALAUnit,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Wrap a CoALAUnit as a pydantic-ai Tool.

    From the LLM's POV: one tool call. Internally the framework runs all
    four CoALA phases — observe parses LLM-supplied args, retrieve
    fetches memory, act produces output, learn records. Output is
    returned to the agent for next-step reasoning.

    Tool name defaults to ``type(unit).__name__``; description defaults
    to the unit's class docstring. Both overridable via kwargs.

    The LLM-facing arg schema is derived from the unit's ``observe``
    signature (specifically, the parameter after ``self``). Apps choose
    the schema by typing ``observe``'s input: ``BaseModel`` for nested
    JSON, primitives for flat args.
    """
    unit_name = name or type(unit).__name__
    unit_desc = description or (type(unit).__doc__ or "").strip() or None

    observe_sig = inspect.signature(type(unit).observe)
    # Drop ``self`` — remaining is the InT parameter
    params_after_self = list(observe_sig.parameters.values())[1:]
    if len(params_after_self) != 1:
        raise ValueError(
            f"CoALAUnit.observe must take exactly one parameter after self; "
            f"{type(unit).__name__}.observe has {len(params_after_self)}",
        )
    input_param = params_after_self[0]

    async def _runner(**kwargs: Any) -> Any:
        input_value = kwargs[input_param.name]
        observation = await unit.observe(input_value)
        context     = await unit.retrieve(observation)
        output      = await unit.act(observation, context)
        await unit.learn(observation, context, output)
        return output

    _runner.__signature__ = inspect.Signature(
        parameters=[inspect.Parameter(
            input_param.name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=input_param.annotation,
        )],
        return_annotation=observe_sig.return_annotation,
    )
    _runner.__name__ = unit_name
    _runner.__doc__  = unit_desc

    return Tool(_runner, name=unit_name, description=unit_desc, takes_ctx=False)


__all__ = ["as_tool"]

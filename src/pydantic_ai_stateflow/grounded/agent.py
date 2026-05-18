from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_stateflow.grounded._spec import OutputSpec
from pydantic_ai_stateflow.grounded.resolver import GroundedResolver

OutT = TypeVar("OutT", bound=BaseModel)


class GroundedResult(BaseModel, Generic[OutT]):
    """Run result, typed as the original OutT for IDE / mypy users.

    The actual model instance at runtime is a `DynamicOutT` with Literal-narrowed
    fields. Users treat it as OutT. Hydration uses `_spec` to walk Ref fields.
    """

    model_config = {"arbitrary_types_allowed": True}

    value: Any           # OutT-typed externally; runtime is DynamicOutT
    raw: Any             # AgentRunResult
    _spec: OutputSpec    # internal; used by Task 18 hydration


class GroundedAgent(Generic[OutT]):
    """Wrapper that builds a per-call dynamic output type and delegates to agent.run."""

    def __init__(self, agent: Agent[Any, OutT], *, output_type: type[OutT]) -> None:
        self.agent = agent
        self.output_type = output_type
        self._resolver = GroundedResolver(output_type)

    async def run(
        self,
        context: BaseModel,
        *,
        instructions: str | None = None,
        constraints: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> GroundedResult[OutT]:
        dynamic_output, spec = self._resolver.build(context, constraints=constraints)

        # Build a fresh Agent that uses the dynamic output_type.
        # We don't mutate `self.agent` to keep it reusable across runs.
        per_call_agent: Agent[Any, Any] = Agent(
            model=self.agent.model,
            output_type=dynamic_output,
        )
        user_prompt = instructions or "Produce output matching the schema."
        run_result = await per_call_agent.run(user_prompt, **agent_kwargs)

        return GroundedResult(value=run_result.output, raw=run_result, _spec=spec)

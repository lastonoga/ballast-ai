from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, PrivateAttr
from pydantic_ai import Agent

from ballast.grounded._spec import OutputSpec
from ballast.grounded.resolver import GroundedResolver
from ballast.grounded.selector import SelectorRegistry

OutT = TypeVar("OutT", bound=BaseModel)


class GroundedResult(BaseModel, Generic[OutT]):
    """Run result, typed as the original OutT for IDE / mypy users.

    The actual model instance at runtime is a `DynamicOutT` with Literal-narrowed
    fields. Users treat it as OutT. Hydration uses `_spec` to walk Ref fields.
    """

    model_config = {"arbitrary_types_allowed": True}

    value: Any           # OutT-typed externally; runtime is DynamicOutT
    raw: Any             # AgentRunResult
    _spec: OutputSpec = PrivateAttr()

    async def hydrate(self, **repos: Any) -> dict[str, Any]:
        """Replace Ref instances in `value` with entities from repos.

        `repos` is keyed by entity class __name__: pass `Item=item_repo, ...`.
        """
        from ballast.grounded.hydration import HydrationMap

        repos_by_type: dict[type, Any] = {}
        for type_name, repo in repos.items():
            for t in self._spec.referenced_entity_types:
                if t.__name__ == type_name:
                    repos_by_type[t] = repo
                    break
        return await HydrationMap(self._spec).hydrate(self.value, repos=repos_by_type)


class GroundedAgent(Generic[OutT]):
    """Wrapper that builds a per-call dynamic output type and delegates to agent.run."""

    def __init__(
        self,
        agent: Agent[Any, OutT],
        *,
        output_type: type[OutT],
        selectors: SelectorRegistry | None = None,
    ) -> None:
        self.agent = agent
        self.output_type = output_type
        self.selectors = selectors
        self._resolver = GroundedResolver(output_type)

    async def run(
        self,
        context: BaseModel,
        *,
        instructions: str | None = None,
        constraints: dict[str, Any] | None = None,
        selector_ctx: Any | None = None,
        **agent_kwargs: Any,
    ) -> GroundedResult[OutT]:
        # ``abuild`` is selector-aware AND backward-compatible with
        # HydrationMap-style context-scan; safe to always use here.
        dynamic_output, spec = await self._resolver.abuild(
            context,
            selector_ctx=selector_ctx if selector_ctx is not None else context,
            selectors=self.selectors,
            constraints=constraints,
        )

        # Build a fresh Agent that uses the dynamic output_type.
        # We don't mutate `self.agent` to keep it reusable across runs.
        per_call_agent: Agent[Any, Any] = Agent(
            model=self.agent.model,
            output_type=dynamic_output,
        )
        user_prompt = instructions or "Produce output matching the schema."
        run_result = await per_call_agent.run(user_prompt, **agent_kwargs)

        result: GroundedResult[OutT] = GroundedResult(value=run_result.output, raw=run_result)
        result._spec = spec
        return result

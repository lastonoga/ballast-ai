from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.errors import StateflowError
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName


class BudgetExhausted(StateflowError):  # noqa: N818
    """Raised by BudgetGuard when the run exceeds the configured budget."""

    code = "STATEFLOW_CAPABILITY_BUDGET_EXHAUSTED"
    status_code = 429

    def __init__(self, reason: str, **details: Any) -> None:
        self.reason = reason
        self.details = details
        super().__init__(
            f"BudgetExhausted: {reason} ({details})",
            hint=(
                "Raise the budget on ``BudgetGuard`` (max_iterations / "
                "max_input_tokens / max_output_tokens) or shorten the run."
            ),
            context={"reason": reason, **details},
        )


class BudgetGuard(StateflowCapability):
    """Outermost capability: refuses runs that exceed iteration / token budget.

    Per-run state is isolated via ``for_run()`` which returns a fresh instance for
    each agent run. This is compatible with DBOS replay because the state lives
    on the capability instance scoped to the run, not on a global object.

    NOTE on ``ctx.state``: pydantic-ai's ``RunContext`` does not expose a ``state``
    dict. State is kept on the per-run instance returned from ``for_run()`` instead.
    """

    name = "budget_guard"

    def __init__(
        self,
        *,
        max_iterations: int = 20,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        # Per-run counters (populated only on the per-run clone returned by for_run)
        self._iterations: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> BudgetGuard:
        """Return a fresh per-run instance so counters are isolated across runs."""
        return BudgetGuard(
            max_iterations=self.max_iterations,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
        )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    @traced(
        TraceName.CAPABILITY_BUDGET_GUARD,
        attrs=lambda self, ctx, request_context: {
            "phase": "before_model_request",
            "iterations": self._iterations,
            "max_iterations": self.max_iterations,
        },
    )
    async def before_model_request(
        self,
        ctx: RunContext[Any],
        # NOTE: positional (not keyword-only) per pydantic-ai 0.0.13+ AbstractCapability signature
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if self._iterations >= self.max_iterations:
            raise BudgetExhausted(
                reason="max_iterations",
                at_step=self._iterations,
                limit=self.max_iterations,
            )
        self._iterations += 1
        return request_context

    @traced(
        TraceName.CAPABILITY_BUDGET_GUARD,
        attrs=lambda self, ctx, *, request_context, response: {
            "phase": "after_model_request",
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        },
    )
    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        usage = response.usage
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        if self.max_input_tokens is not None and self._input_tokens > self.max_input_tokens:
            raise BudgetExhausted(
                reason="max_input_tokens",
                consumed=self._input_tokens,
                limit=self.max_input_tokens,
            )
        if self.max_output_tokens is not None and self._output_tokens > self.max_output_tokens:
            raise BudgetExhausted(
                reason="max_output_tokens",
                consumed=self._output_tokens,
                limit=self.max_output_tokens,
            )
        return response

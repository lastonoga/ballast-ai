from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import OutputContext, RawOutput

from pydantic_ai_stateflow.capabilities.base import StateflowCapability


def _build_feedback(error: ValidationError, raw_output: Any) -> str:
    """Translate a Pydantic ValidationError into a model-friendly retry hint.

    Special-cases:
    - ``literal_error``: list allowed values + actual bad value
    - ``missing``: name the required field
    - default: pass through Pydantic's message with the field path
    """
    msgs: list[str] = []
    for err in error.errors():
        loc = ".".join(str(part) for part in err["loc"])
        etype = err["type"]
        if etype == "literal_error":
            allowed = err.get("ctx", {}).get("expected", "")
            actual = err.get("input")
            msgs.append(
                f"Field '{loc}' must be one of: {allowed}. You returned: {actual!r}."
            )
        elif etype == "missing":
            msgs.append(f"Required field '{loc}' is missing.")
        else:
            msgs.append(f"{loc}: {err['msg']}")
    return "Output validation failed:\n" + "\n".join(f"- {m}" for m in msgs)


class GroundedRetry(StateflowCapability):
    """Converts Pydantic validation errors on output into structured ModelRetry.

    Gives the model precise feedback (which field, what was expected, what
    it returned) instead of a generic "JSON invalid" message. From the
    spec: this lifts F1 from ~0.84 to ~0.96 in structured-output tasks.

    Per-run state (attempt counter) is isolated via ``for_run()`` which returns
    a fresh instance for each agent run. ``RunContext`` does not expose a
    ``state`` dict in pydantic-ai, so attempts live on the per-run clone.
    """

    name = "grounded_retry"

    def __init__(self, *, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._attempts: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> GroundedRetry:
        return GroundedRetry(max_retries=self.max_retries)

    async def on_output_validate_error(
        self,
        ctx: RunContext[Any],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:
        # Only translate Pydantic ValidationError; pass ModelRetry through unchanged.
        if not isinstance(error, ValidationError):
            raise error
        if self._attempts >= self.max_retries:
            raise error
        self._attempts += 1
        raise ModelRetry(_build_feedback(error, output))

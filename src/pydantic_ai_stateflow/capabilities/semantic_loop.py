from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.capabilities.helpers import Embedder, SemanticDeduper


def _default_response_text(response: ModelResponse) -> str:
    """Concatenate TextParts and serialise ToolCallPart args as stable JSON.

    Stable JSON keys ensure identical tool calls — regardless of arg dict
    iteration order — produce identical snapshot strings.
    """
    bits: list[str] = []
    for p in response.parts:
        if isinstance(p, TextPart):
            bits.append(p.content)
        elif isinstance(p, ToolCallPart):
            args = p.args if isinstance(p.args, dict) else {"_raw": p.args}
            bits.append(f"{p.tool_name}({json.dumps(args, sort_keys=True, default=str)})")
    return "\n".join(bits)


class SemanticLoopDetector(StateflowCapability):
    """Detects repeated model responses within a single agent.run().

    Works at the model-response level (raw text + tool-call args). For
    detecting loops between Pattern iterations on TYPED output, see
    `TypedLoopGuard`.

    Per-run state is isolated via ``for_run()`` which returns a fresh clone
    owning its own ``SemanticDeduper``. The base instance holds only config
    (embedder + thresholds) and never accumulates history across runs.
    """

    name = "semantic_loop_detector"

    def __init__(
        self,
        *,
        embedder: Embedder,
        threshold: float = 0.95,
        window: int = 3,
        selector: Callable[[ModelResponse], str] = _default_response_text,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.window = window
        self.selector = selector
        self._deduper: SemanticDeduper | None = None

    async def for_run(self, ctx: RunContext[Any]) -> SemanticLoopDetector:
        """Return a fresh per-run instance with an isolated SemanticDeduper."""
        clone = SemanticLoopDetector(
            embedder=self.embedder,
            threshold=self.threshold,
            window=self.window,
            selector=self.selector,
        )
        clone._deduper = SemanticDeduper(self.embedder)
        return clone

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        assert self._deduper is not None, "after_model_request called on base instance; for_run() must run first"
        snapshot = self.selector(response)
        await self._deduper.add_and_check(
            snapshot, threshold=self.threshold, window=self.window
        )
        return response

"""Built-in ``TraceWindow`` implementations.

Apps choose how much message history the judge sees:

- ``FullTrace`` — entire history (precise, expensive on long sessions).
- ``LastNMessages(n)`` — tail.
- ``SinceLastUserMessage()`` — from the most recent user message onward.
- ``TokenBudgetWindow(max_tokens)`` — trim from the head until total
  approximate token count fits the cap.
"""
from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from ballast.drift._protocols import DriftContext


class FullTrace:
    """Return every message."""

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        return list(ctx.messages)


class LastNMessages:
    """Return the last ``n`` messages (entire history if shorter)."""

    def __init__(self, n: int = 10) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        return list(ctx.messages[-self._n :])


class SinceLastUserMessage:
    """Slice from the most recent user prompt onward."""

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        # Walk backwards looking for a ModelRequest whose parts contain
        # at least one UserPromptPart. Return slice from that index.
        for i in range(len(ctx.messages) - 1, -1, -1):
            msg = ctx.messages[i]
            if isinstance(msg, ModelRequest) and any(
                isinstance(p, UserPromptPart) for p in msg.parts
            ):
                return list(ctx.messages[i:])
        return list(ctx.messages)


class TokenBudgetWindow:
    """Trim history from the head until total token estimate fits ``max_tokens``.

    Approximation: sum of ``len(part.content)`` for all parts that have a
    ``content`` attribute, divided by 4 (rough English rule-of-thumb), with a
    minimum of 1 token per message.  Content-only estimation avoids being
    misled by verbose pydantic-ai repr strings.
    """

    def __init__(self, max_tokens: int = 4000) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max = max_tokens

    @staticmethod
    def _cost(msg: ModelMessage) -> int:
        content_len = sum(
            len(getattr(p, "content", "")) for p in msg.parts
        )
        return max(1, content_len // 4)

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        if not ctx.messages:
            return []
        tail: list[ModelMessage] = []
        budget = self._max
        for msg in reversed(ctx.messages):
            cost = self._cost(msg)
            if cost > budget:
                break
            tail.append(msg)
            budget -= cost
        tail.reverse()
        return tail


__all__ = [
    "FullTrace",
    "LastNMessages",
    "SinceLastUserMessage",
    "TokenBudgetWindow",
]

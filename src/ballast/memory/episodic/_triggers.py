"""``RememberTurn`` — capability that writes episodes after successful turns.

Default gate: always-True (apps wire a callable returning ``False`` to
skip, e.g. ``gate=lambda ctx, result: judge_passed(result)`` — typical
integration with LLMJudge).
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic_ai import RunContext

from ballast.capabilities.base import BallastCapability
from ballast.memory._scope import Scope
from ballast.memory.episodic._facade import EpisodicMemory
from ballast.memory.episodic._models import Episode

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult


_log = logging.getLogger("ballast.remember_turn")


GateFn = Callable[[RunContext[Any], Any], bool | Awaitable[bool]]
SummarizerFn = Callable[[RunContext[Any], Any], Awaitable[str]]


class RememberTurn(BallastCapability):
    """After each agent run, if the gate passes, summarize + persist."""

    name = "remember_turn"

    def __init__(
        self,
        *,
        memory: EpisodicMemory,
        gate: GateFn | None = None,
        summarizer: SummarizerFn | None = None,
    ) -> None:
        self._memory = memory
        self._gate = gate or (lambda *_: True)
        self._summarizer = summarizer or self._default_summarizer

    @staticmethod
    async def _default_summarizer(ctx: RunContext[Any], result: Any) -> str:
        # Take whatever string/repr is on result.output.
        output = getattr(result, "output", "") or ""
        return str(output)[:400]

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: "AgentRunResult[Any]",
    ) -> "AgentRunResult[Any]":
        try:
            gate_out = self._gate(ctx, result)
            passed = await gate_out if hasattr(gate_out, "__await__") else gate_out
            if not passed:
                return result
            summary = await self._summarizer(ctx, result)
            user_id = getattr(getattr(ctx, "deps", None), "user_id", None)
            thread_id = getattr(getattr(ctx, "deps", None), "parent_thread_id", None)
            ep = Episode(
                id=str(uuid4()),
                source="remember-turn",
                occurred_at=datetime.now(timezone.utc),
                scope=Scope(
                    user_id=user_id,
                    thread_id=str(thread_id) if thread_id is not None else None,
                ),
                preview=summary[:200],
                summary=summary,
            )
            await self._memory.remember(ep)
        except Exception:
            _log.exception("RememberTurn after_run failed (swallowed)")
        return result


__all__ = ["RememberTurn"]

"""``EpisodicMemory`` facade — federation + dual surface (push/pull)."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import Episode, RecallResult
from ballast.memory.episodic._protocol import EpisodicSource
from ballast.memory.episodic.strategies._protocol import RecallStrategy
from ballast.memory.episodic.strategies._topk import TopK

_log = logging.getLogger(__name__)


class EpisodicMemory:
    """Federation of EpisodicSource impls. Two surfaces:

    - Direct (workflow push):  ``await memory.episodic_for(intent=…)``
    - Tool (agent pull):       ``memory.as_tools()`` returns pydantic-ai tools
    """

    def __init__(
        self,
        sources: list[EpisodicSource],
        *,
        default_strategy: RecallStrategy | None = None,
        default_scope_builder: Callable[[], Scope] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("EpisodicMemory requires at least one source")
        self._sources = sources
        self._default_strategy = default_strategy or TopK()
        self._default_scope_builder = default_scope_builder

    async def episodic_for(
        self,
        *,
        intent: str,
        strategy: RecallStrategy | None = None,
        scope: Scope | None = None,
    ) -> RecallResult:
        used_strategy = strategy or self._default_strategy
        used_scope = (
            scope if scope is not None
            else (
                self._default_scope_builder() if self._default_scope_builder
                else Scope()
            )
        )
        result = await used_strategy.execute(
            intent=intent, sources=self._sources, scope=used_scope,
        )
        if getattr(used_strategy, "requires_grounding", False) and not result.references:
            _log.warning(
                "episodic recall(intent=%r) returned 0 references but strategy "
                "requires_grounding=True — output_type with Ref[T] will fail",
                intent,
            )
        return result

    async def remember(self, episode: Episode) -> None:
        async def _safe(src):
            try:
                await src.remember(episode)
            except NotImplementedError:
                pass
            except Exception:
                _log.exception("episodic source %s remember failed", src.name)
        await asyncio.gather(*(_safe(s) for s in self._sources))

    def as_tools(self) -> list:
        from ballast.memory.episodic._tools import build_recall_tool  # noqa: PLC0415
        return [build_recall_tool(self)]


__all__ = ["EpisodicMemory"]

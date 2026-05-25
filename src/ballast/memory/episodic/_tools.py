"""Agent pull surface — exposes ``EpisodicMemory`` as a pydantic-ai tool."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Tool

if TYPE_CHECKING:
    from ballast.memory.episodic._facade import EpisodicMemory


def build_recall_tool(memory: "EpisodicMemory") -> Tool:
    """Return a pydantic-ai Tool wrapping ``memory.episodic_for``."""

    async def recall_episodes(
        intent: str,
        k: int = 5,
    ) -> list[dict]:
        """Recall episodes from your past activity that are relevant to
        the given intent. Returns up to ``k`` episodes with id / preview /
        occurred_at. Use this when the user references prior work or
        when you suspect you've handled a similar task before."""
        from ballast.memory.episodic.strategies._topk import TopK  # noqa: PLC0415
        result = await memory.episodic_for(intent=intent, strategy=TopK(k=k))
        return [
            {
                "id":          se.episode.id,
                "source":      se.episode.source,
                "preview":     se.episode.preview,
                "summary":     se.episode.summary,
                "occurred_at": se.episode.occurred_at.isoformat(),
                "score":       se.score,
            }
            for se in result.episodes
        ]

    return Tool(recall_episodes, takes_ctx=False)


__all__ = ["build_recall_tool"]

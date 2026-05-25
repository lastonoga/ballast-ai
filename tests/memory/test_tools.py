"""``EpisodicMemory.as_tools()`` — pydantic-ai-compatible recall tool."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, EpisodicMemory, ScoredEpisode


class _FakeSource:
    name = "fake"
    def __init__(self, returns): self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


@pytest.mark.asyncio
async def test_as_tools_returns_one_recall_tool() -> None:
    src = _FakeSource([ScoredEpisode(
        episode=Episode(id="a", source="fake",
                        occurred_at=datetime.now(UTC),
                        scope=Scope(), preview="hello"),
        score=0.9,
    )])
    mem = EpisodicMemory(sources=[src])
    tools = mem.as_tools()
    assert len(tools) == 1
    # The tool function should be async and accept (intent, k).
    # pydantic-ai Tool exposes its callable via .function attribute.
    out = await tools[0].function(intent="x", k=3)
    # Returns a list of dicts (JSON-serializable summary) for agent inspection.
    assert isinstance(out, list) and out[0]["preview"] == "hello"

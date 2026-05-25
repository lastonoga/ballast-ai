"""``SemanticMemory`` facade — thin tool aggregator + collision detection."""
from __future__ import annotations

import pytest

from ballast.memory.semantic import (
    DomainSemanticSource,
    SemanticMemory,
    memory_tool,
)


class _NotesSource(DomainSemanticSource):
    name = "notes"

    @memory_tool
    async def find_by_tag(self, tag: str) -> list[str]:
        """Find notes by tag."""
        return [tag]


class _OrdersSource(DomainSemanticSource):
    name = "orders"

    @memory_tool
    async def recent(self, days: int = 7) -> list[str]:
        """Recent orders."""
        return [f"day{i}" for i in range(days)]


class _CollidingSource(DomainSemanticSource):
    name = "colliding"

    @memory_tool
    async def find_by_tag(self, tag: str) -> list[str]:
        """Same tool name as _NotesSource — should collide."""
        return [tag]


def test_aggregates_tools_across_sources() -> None:
    memory = SemanticMemory(sources=[_NotesSource(), _OrdersSource()])
    tool_names = {t.name for t in memory.as_tools()}
    assert tool_names == {"find_by_tag", "recent"}


def test_collision_raises_on_construction() -> None:
    with pytest.raises(ValueError, match="collision"):
        SemanticMemory(sources=[_NotesSource(), _CollidingSource()])


def test_collision_message_mentions_both_sources() -> None:
    with pytest.raises(ValueError) as exc_info:
        SemanticMemory(sources=[_NotesSource(), _CollidingSource()])
    msg = str(exc_info.value)
    assert "notes" in msg and "colliding" in msg
    assert "find_by_tag" in msg


def test_collision_can_be_resolved_by_name_override() -> None:
    class _Resolved(DomainSemanticSource):
        name = "resolved"

        @memory_tool(name="alt_find_by_tag")
        async def find_by_tag(self, tag: str) -> list[str]:
            return [tag]

    memory = SemanticMemory(sources=[_NotesSource(), _Resolved()])
    tool_names = {t.name for t in memory.as_tools()}
    assert tool_names == {"find_by_tag", "alt_find_by_tag"}


def test_empty_sources_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SemanticMemory(sources=[])


def test_list_sources_returns_registered() -> None:
    src1, src2 = _NotesSource(), _OrdersSource()
    memory = SemanticMemory(sources=[src1, src2])
    assert memory.list_sources() == [src1, src2]

"""``extract_memory_tools`` — introspection from SemanticSource → pydantic-ai Tools."""
from __future__ import annotations

from ballast.memory.semantic import DomainSemanticSource, memory_tool
from ballast.memory.semantic._tools import extract_memory_tools


class _SampleSource(DomainSemanticSource):
    name = "sample"

    @memory_tool
    async def find_by_tag(self, tag: str, limit: int = 10) -> list[str]:
        """Find samples tagged with `tag`. Most recent first."""
        return [f"{tag}:{i}" for i in range(limit)]

    @memory_tool(name="recent_samples")
    async def recent(self, days: int = 7) -> list[str]:
        """Recent samples."""
        return [f"day{i}" for i in range(days)]

    async def _internal_helper(self) -> None:
        """Not decorated — must not be exposed."""
        return None

    async def public_undecorated(self) -> str:
        """Public method without decorator — must not be exposed."""
        return "x"


def test_extracts_only_decorated_methods() -> None:
    tools = extract_memory_tools(_SampleSource())
    tool_names = {t.name for t in tools}
    assert tool_names == {"find_by_tag", "recent_samples"}


def test_uses_override_name_when_set() -> None:
    tools = extract_memory_tools(_SampleSource())
    name_method = next(t for t in tools if t.name == "recent_samples")
    # `name=` override wins; method's bare attribute name is "recent".
    assert name_method.name == "recent_samples"


def test_uses_docstring_as_description_fallback() -> None:
    tools = extract_memory_tools(_SampleSource())
    find_tool = next(t for t in tools if t.name == "find_by_tag")
    assert "tagged with" in (find_tool.description or "").lower()


def test_returns_empty_for_source_with_no_decorated_methods() -> None:
    class _Empty(DomainSemanticSource):
        name = "empty"
        async def helper(self) -> None: ...
    assert extract_memory_tools(_Empty()) == []

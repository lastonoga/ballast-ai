"""``SemanticMemory`` — thin tool aggregator for agent-pull exposure."""
from __future__ import annotations

from pydantic_ai import Tool

from ballast.memory.semantic._protocol import SemanticSource
from ballast.memory.semantic._tools import extract_memory_tools


class SemanticMemory:
    """Federation of ``SemanticSource`` impls for agent-pull tool exposure.

    Workflow code imports + calls source singletons directly — no
    facade indirection. This object exists purely to:
      1. collect tools from all sources for ``Agent(tools=...)``,
      2. validate no tool-name collisions across sources at construction,
      3. provide introspection (``list_sources``) for admin / debug.
    """

    def __init__(self, sources: list[SemanticSource]) -> None:
        if not sources:
            raise ValueError("SemanticMemory requires at least one source")
        self._sources = sources
        self._validate_no_collisions()

    def as_tools(self) -> list[Tool]:
        """Return pydantic-ai ``Tool`` instances collected from all
        ``@memory_tool``-decorated methods across all sources."""
        return [
            tool
            for src in self._sources
            for tool in extract_memory_tools(src)
        ]

    def list_sources(self) -> list[SemanticSource]:
        """Return the registered sources (for introspection / debug)."""
        return list(self._sources)

    def _validate_no_collisions(self) -> None:
        """Raise ``ValueError`` if two sources expose the same tool name.

        Apps fix this by passing ``@memory_tool(name=...)`` on the
        method whose tool name should change.
        """
        owner_by_tool: dict[str, str] = {}
        for src in self._sources:
            for tool in extract_memory_tools(src):
                if tool.name in owner_by_tool:
                    raise ValueError(
                        f"SemanticMemory tool-name collision: {tool.name!r} "
                        f"defined by both {owner_by_tool[tool.name]!r} and "
                        f"{src.name!r}. Use @memory_tool(name=...) to "
                        "disambiguate."
                    )
                owner_by_tool[tool.name] = src.name


__all__ = ["SemanticMemory"]

"""``SemanticSource`` Protocol — typed source of structured facts."""
from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class SemanticSource(Protocol):
    """Source of typed structured facts about the domain.

    Implementations expose one or more ``@memory_tool``-decorated async
    methods. ``SemanticMemory.as_tools()`` introspects each registered
    source and builds pydantic-ai ``Tool`` instances from the marked
    methods. The framework knows nothing about the methods' signatures
    or return types — they flow through to the LLM as typed tools.

    No abstract methods (unlike ``EpisodicSource``) — semantic facts
    have wildly different per-domain shapes; only the ``name`` attribute
    is required.
    """

    name: ClassVar[str]


__all__ = ["SemanticSource"]

"""``@memory_tool`` — marks async methods on a SemanticSource for
exposure to the agent via ``SemanticMemory.as_tools()``."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


@overload
def memory_tool(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def memory_tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def memory_tool(
    fn: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Mark an async method on a ``SemanticSource`` as a tool exposed
    to the agent.

    Bare form::

        @memory_tool
        async def find_by_tag(self, tag: str, limit: int = 10) -> list[Note]:
            \"\"\"Find notes tagged with `tag`. Most recent first.\"\"\"
            ...

    With overrides::

        @memory_tool(name="search_notes_by_tag",
                     description="Search notes by tag")
        async def find_by_tag(self, tag: str): ...

    The decorator does not wrap the callable — it attaches marker
    attributes (``__memory_tool__`` + optional overrides) that
    ``extract_memory_tools`` reads during introspection.
    """
    def decorate(f: Callable[P, R]) -> Callable[P, R]:
        f.__memory_tool__ = True                    # type: ignore[attr-defined]
        f.__memory_tool_name__ = name               # type: ignore[attr-defined]
        f.__memory_tool_description__ = description # type: ignore[attr-defined]
        return f

    if fn is not None:
        return decorate(fn)
    return decorate


__all__ = ["memory_tool"]

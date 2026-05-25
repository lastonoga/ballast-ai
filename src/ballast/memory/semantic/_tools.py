"""Introspection helpers — turn ``@memory_tool``-decorated methods on a
``SemanticSource`` into pydantic-ai ``Tool`` instances."""
from __future__ import annotations

import inspect

from pydantic_ai import Tool

from ballast.memory.semantic._protocol import SemanticSource


def extract_memory_tools(source: SemanticSource) -> list[Tool]:
    """Find every ``@memory_tool``-marked async method on ``source``
    and wrap each in a pydantic-ai ``Tool``.

    Tool name resolution: ``@memory_tool(name=...)`` override wins;
    otherwise the method's attribute name is used.

    Description resolution: ``@memory_tool(description=...)`` override
    wins; otherwise the method's docstring (stripped) is used; if both
    are absent, ``None``.

    Argument schema: derived by pydantic-ai from ``inspect.signature``
    + the method's type hints. The framework adds no extra wrapping.
    """
    tools: list[Tool] = []
    for attr_name, attr_value in inspect.getmembers(source, inspect.iscoroutinefunction):
        if not getattr(attr_value, "__memory_tool__", False):
            continue
        tool_name = (
            getattr(attr_value, "__memory_tool_name__", None) or attr_name
        )
        override_description = getattr(attr_value, "__memory_tool_description__", None)
        if override_description:
            description = override_description
        elif attr_value.__doc__:
            description = attr_value.__doc__.strip()
        else:
            description = None
        tools.append(Tool(
            attr_value,
            name=tool_name,
            description=description,
            takes_ctx=False,
        ))
    return tools


__all__ = ["extract_memory_tools"]

"""``@memory_tool`` decorator — marker for tool-exposable methods."""
from __future__ import annotations

import pytest

from ballast.memory.semantic import memory_tool


def test_bare_decorator_marks_function() -> None:
    @memory_tool
    async def my_method(self, x: int) -> str:
        """My docstring."""
        return str(x)

    assert getattr(my_method, "__memory_tool__", False) is True
    assert getattr(my_method, "__memory_tool_name__", None) is None
    assert getattr(my_method, "__memory_tool_description__", None) is None


def test_decorator_with_name_override() -> None:
    @memory_tool(name="custom_name")
    async def my_method(self, x: int) -> str:
        return str(x)

    assert my_method.__memory_tool__ is True
    assert my_method.__memory_tool_name__ == "custom_name"


def test_decorator_with_description_override() -> None:
    @memory_tool(description="Custom description")
    async def my_method(self) -> str:
        return "x"

    assert my_method.__memory_tool__ is True
    assert my_method.__memory_tool_description__ == "Custom description"


def test_decorator_preserves_callable_behavior() -> None:
    """Decorator doesn't wrap — calling the method works normally."""
    import asyncio

    @memory_tool
    async def doubler(x: int) -> int:
        return x * 2

    out = asyncio.run(doubler(5))
    assert out == 10


def test_decorator_with_both_overrides() -> None:
    @memory_tool(name="foo", description="bar")
    async def my_method(self) -> None: ...

    assert my_method.__memory_tool_name__ == "foo"
    assert my_method.__memory_tool_description__ == "bar"

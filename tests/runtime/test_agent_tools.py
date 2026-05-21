"""``@StateflowAgent.tool`` decorator + auto-grounded ``prepare`` hooks.

NOTE: this module intentionally does NOT use
``from __future__ import annotations``. pydantic-ai's tool registration
introspects parameter types via ``get_type_hints(fn, include_extras=True)``
at decoration time, and lazy-evaluated annotations only resolve through
the function's ``__globals__`` — string annotations defined inside a
test function can't see imports made in the test body. Same constraint
that applies to real agent files; see ``notes_app.agent``.
"""

from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID, uuid4

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pydantic_ai_stateflow.grounded import Ref, Selector
from pydantic_ai_stateflow.runtime import StateflowAgent


# ── Toolly: mixed ctx and plain tools ────────────────────────────────────────


class _Toolly(StateflowAgent):
    name = "toolly"

    def build_agent(self) -> Agent[None, str]:
        return Agent(TestModel(custom_output_text="ok"), output_type=str)

    async def build_deps(self, **_kw: Any) -> None:
        return None


@_Toolly.tool
def with_ctx(ctx: RunContext[None], x: int) -> int:
    del ctx
    return x


@_Toolly.tool(retries=3)
def with_ctx_and_kwargs(ctx: RunContext[None]) -> str:
    del ctx
    return "ok"


@_Toolly.tool
def plain_no_ctx(y: str) -> str:
    return y


def test_tool_decorator_registers_on_class() -> None:
    """``@SomeAgent.tool`` lands every tool on the underlying Agent."""
    instance = _Toolly()
    a = instance.agent
    tools = a._function_toolset.tools  # noqa: SLF001
    assert set(tools) == {"with_ctx", "with_ctx_and_kwargs", "plain_no_ctx"}
    assert tools["with_ctx_and_kwargs"].max_retries == 3


def test_tool_decorator_runs_idempotently() -> None:
    """Touching ``self.agent`` twice doesn't double-register tools."""
    instance = _Toolly()
    a1 = instance.agent
    a2 = instance.agent
    assert a1 is a2  # cached_property
    tools = a1._function_toolset.tools  # noqa: SLF001
    assert len(tools) == 3


# ── Parent/Child inheritance ────────────────────────────────────────────────


class _Parent(StateflowAgent):
    name = "parent"

    def build_agent(self) -> Agent[None, str]:
        return Agent(TestModel(), output_type=str)

    async def build_deps(self, **_kw: Any) -> None:
        return None


@_Parent.tool
def parent_tool() -> str:
    return "p"


class _Child(_Parent):
    name = "child"


@_Child.tool
def child_tool() -> str:
    return "c"


def test_subclass_inherits_parent_tools() -> None:
    instance = _Child()
    tools = instance.agent._function_toolset.tools  # noqa: SLF001
    assert set(tools) == {"parent_tool", "child_tool"}


def test_parent_in_isolation_has_only_own_tools() -> None:
    parent_instance = _Parent()
    parent_tools = parent_instance.agent._function_toolset.tools  # noqa: SLF001
    assert set(parent_tools) == {"parent_tool"}


# ── Subclass override of a parent tool ──────────────────────────────────────


class _Base(StateflowAgent):
    name = "base"

    def build_agent(self) -> Agent[None, str]:
        return Agent(TestModel(), output_type=str)

    async def build_deps(self, **_kw: Any) -> None:
        return None


@_Base.tool
def shared() -> str:
    return "from_base"


class _Sub(_Base):
    name = "sub"


@_Sub.tool
def shared() -> str:  # noqa: F811 — intentional override
    return "from_sub"


def test_subclass_override_replaces_parent_tool() -> None:
    """A subclass tool with the same Python name shadows the parent's."""
    sub_tools = _Sub().agent._function_toolset.tools  # noqa: SLF001
    assert list(sub_tools) == ["shared"]
    assert sub_tools["shared"].function() == "from_sub"

    # Parent in isolation still gets its own.
    base_tools = _Base().agent._function_toolset.tools  # noqa: SLF001
    assert base_tools["shared"].function() == "from_base"


# ── Auto-grounded ``prepare`` hook installation ─────────────────────────────


@dataclass
class _Item:
    id: UUID
    name: str


class _ItemDeps:
    def __init__(self) -> None:
        self.items: list[_Item] = []


async def _list_items(c: Any) -> list[_Item]:
    return list(c.deps.items)


class _ItemAgent(StateflowAgent):
    name = "items"

    def build_agent(self) -> Agent[_ItemDeps, str]:
        return Agent(TestModel(), output_type=str, deps_type=_ItemDeps)

    async def build_deps(self, **_kw: Any) -> _ItemDeps:
        return _ItemDeps()


@_ItemAgent.tool
async def pick(
    ctx: RunContext[_ItemDeps],
    item_id: Annotated[Ref[_Item], Selector(_list_items)],
) -> str:
    del ctx
    return str(item_id.id if isinstance(item_id, Ref) else item_id)


def test_auto_grounded_prepare_hook_installed() -> None:
    """``Annotated[Ref[T], Selector(...)]`` gets a ``prepare`` hook
    installed without any explicit ``register_grounded_tools`` call.
    """
    instance = _ItemAgent()
    pick_tool = instance.agent._function_toolset.tools["pick"]  # noqa: SLF001
    assert pick_tool.prepare is not None
    # The framework tags the installed prepare so re-touching ``agent``
    # would be idempotent — sanity-check the marker.
    assert getattr(pick_tool, "_grounded_prepare_installed", False) is True


async def test_auto_grounded_prepare_narrows_schema_to_enum() -> None:
    """The auto-installed prepare hook narrows the param JSON schema
    to a closed enum of existing item ids at run-time."""
    instance = _ItemAgent()
    pick_tool = instance.agent._function_toolset.tools["pick"]  # noqa: SLF001

    deps = _ItemDeps()
    deps.items = [_Item(id=uuid4(), name="a"), _Item(id=uuid4(), name="b")]

    @dataclass
    class _Ctx:
        deps: _ItemDeps

    new_def = await pick_tool.prepare(_Ctx(deps=deps), pick_tool.tool_def)  # type: ignore[arg-type, misc]
    assert new_def is not None
    enum_vals = new_def.parameters_json_schema["properties"]["item_id"]["enum"]
    assert set(enum_vals) == {str(i.id) for i in deps.items}

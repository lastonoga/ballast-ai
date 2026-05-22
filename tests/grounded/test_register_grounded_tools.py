"""Tests for ``register_grounded_tools`` — Annotated[Ref[T], Selector(...)]
on pydantic-ai tool parameters narrows the JSON Schema to a closed enum
at run-time (or hides the tool when the set is empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel
from pydantic_ai import Agent

from ballast.grounded.ref import Ref
from ballast.grounded.selector import Selector, SelectorRegistry
from ballast.grounded.tools import register_grounded_tools


class _Note(BaseModel):
    id: UUID
    title: str


@dataclass
class _Deps:
    notes: list[_Note]


def _build_agent_with_inline_selector() -> Agent[_Deps, str]:
    agent: Agent[_Deps, str] = Agent(model="test", deps_type=_Deps, output_type=str)

    @agent.tool
    async def delete_note(
        ctx,  # type: ignore[no-untyped-def]
        note_id: Annotated[Ref[_Note], Selector(lambda c: c.deps.notes)],
    ) -> str:
        return f"deleted {note_id}"

    return agent


async def _run_prepare(agent: Agent[_Deps, str], deps: _Deps) -> object:
    tool = agent._function_toolset.tools["delete_note"]  # noqa: SLF001
    # Stub a RunContext-shaped object — prepare only reads .deps in our
    # selectors, so a minimal dataclass suffices.
    class _Ctx:
        def __init__(self, d: _Deps) -> None:
            self.deps = d

    assert tool.prepare is not None
    return await tool.prepare(_Ctx(deps), tool.tool_def)


async def test_selector_narrows_schema_enum() -> None:
    agent = _build_agent_with_inline_selector()
    register_grounded_tools(agent)

    n1 = _Note(id=uuid4(), title="a")
    n2 = _Note(id=uuid4(), title="b")
    deps = _Deps(notes=[n1, n2])
    new_def = await _run_prepare(agent, deps)

    assert new_def is not None
    schema = new_def.parameters_json_schema  # type: ignore[attr-defined]
    assert schema["properties"]["note_id"]["enum"] == [str(n1.id), str(n2.id)]


async def test_selector_hides_tool_when_empty() -> None:
    agent = _build_agent_with_inline_selector()
    register_grounded_tools(agent)

    new_def = await _run_prepare(agent, _Deps(notes=[]))
    assert new_def is None  # tool hidden


async def test_register_grounded_tools_is_idempotent() -> None:
    agent = _build_agent_with_inline_selector()
    register_grounded_tools(agent)
    first = agent._function_toolset.tools["delete_note"].prepare  # noqa: SLF001

    # Second call must not re-wrap (same prepare callable identity).
    register_grounded_tools(agent)
    second = agent._function_toolset.tools["delete_note"].prepare  # noqa: SLF001
    assert first is second


async def test_named_selector_via_registry() -> None:
    reg = SelectorRegistry()

    agent: Agent[_Deps, str] = Agent(model="test", deps_type=_Deps, output_type=str)

    @agent.tool
    async def archive_note(
        ctx,  # type: ignore[no-untyped-def]
        note_id: Annotated[Ref[_Note], Selector("open_only")],
    ) -> str:
        return "ok"

    n1 = _Note(id=uuid4(), title="open")
    reg.register("open_only", lambda c: c.deps.notes)
    register_grounded_tools(agent, selectors=reg)

    class _Ctx:
        deps = _Deps(notes=[n1])

    tool = agent._function_toolset.tools["archive_note"]  # noqa: SLF001
    new_def = await tool.prepare(_Ctx(), tool.tool_def)  # type: ignore[misc]
    assert new_def is not None
    assert new_def.parameters_json_schema["properties"]["note_id"]["enum"] == [str(n1.id)]


async def test_chains_with_preexisting_prepare() -> None:
    """If a tool already has a ``prepare`` callback, our wrapper chains it."""
    calls: list[str] = []

    async def existing_prepare(_ctx, tool_def):  # type: ignore[no-untyped-def]
        calls.append("existing")
        return tool_def  # pass through unchanged

    agent: Agent[_Deps, str] = Agent(model="test", deps_type=_Deps, output_type=str)

    @agent.tool(prepare=existing_prepare)
    async def delete_note(
        ctx,  # type: ignore[no-untyped-def]
        note_id: Annotated[Ref[_Note], Selector(lambda c: c.deps.notes)],
    ) -> str:
        return "ok"

    register_grounded_tools(agent)

    n1 = _Note(id=uuid4(), title="a")

    class _Ctx:
        deps = _Deps(notes=[n1])

    tool = agent._function_toolset.tools["delete_note"]  # noqa: SLF001
    new_def = await tool.prepare(_Ctx(), tool.tool_def)  # type: ignore[misc]
    assert calls == ["existing"]
    assert new_def is not None
    assert new_def.parameters_json_schema["properties"]["note_id"]["enum"] == [str(n1.id)]


async def test_existing_prepare_returning_none_short_circuits() -> None:
    async def hides(_ctx, _tool_def):  # type: ignore[no-untyped-def]
        return None

    agent: Agent[_Deps, str] = Agent(model="test", deps_type=_Deps, output_type=str)

    @agent.tool(prepare=hides)
    async def delete_note(
        ctx,  # type: ignore[no-untyped-def]
        note_id: Annotated[Ref[_Note], Selector(lambda c: [uuid4()])],
    ) -> str:
        return "ok"

    register_grounded_tools(agent)

    class _Ctx:
        deps = _Deps(notes=[])

    tool = agent._function_toolset.tools["delete_note"]  # noqa: SLF001
    assert (await tool.prepare(_Ctx(), tool.tool_def)) is None  # type: ignore[misc]


async def test_tool_without_ref_params_is_left_alone() -> None:
    agent: Agent[_Deps, str] = Agent(model="test", deps_type=_Deps, output_type=str)

    @agent.tool
    async def plain(ctx, text: str) -> str:  # type: ignore[no-untyped-def]
        return text

    register_grounded_tools(agent)
    assert agent._function_toolset.tools["plain"].prepare is None  # noqa: SLF001

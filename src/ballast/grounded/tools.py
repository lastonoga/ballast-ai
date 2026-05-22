"""``register_grounded_tools`` — scan a pydantic-ai ``Agent`` for tool
parameters annotated with ``Annotated[Ref[T], Selector(...)]`` and install
``prepare=`` hooks that narrow each tool's JSON Schema to a closed enum of
real IDs at run-time.

This is the input-side analog of ``GroundedAgent``'s output-side scanner.
Together they let a single ``Annotated[Ref[T], Selector(...)]`` declaration
constrain BOTH a tool's parameter AND an output field.

Behavior:

1. Resolves each ``Selector`` per-run via the ``SelectorRegistry`` (for
   named selectors) or directly (for inline lambdas).
2. Calls the selector with the run's ``RunContext`` to get a list of UUIDs
   (or entity instances — ``.id`` is extracted).
3. Mutates ``tool_def.parameters_json_schema`` to add ``enum`` and a
   description preview.
4. Returns ``None`` to HIDE the tool when the closed set is empty.

Idempotent: calling twice on the same Agent installs each ``prepare`` only
once (we tag the installed prepare with ``_grounded_prepare_marker``).

Chaining: if a tool already has a ``prepare=`` callback set, we wrap it —
the existing prepare runs first; if it returns ``None`` (hiding) we honor
that immediately; otherwise we apply the Selector layer on top.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace as dataclasses_replace
from typing import Any, get_type_hints

from pydantic_ai import Agent
from pydantic_ai.tools import Tool, ToolDefinition

from ballast.grounded.hydration import HydrationMap
from ballast.grounded.selector import (
    Selector,
    SelectorRegistry,
    extract_selector,
    resolve_selector_ids,
)

_GROUNDED_PREPARE_MARKER = "_grounded_prepare_installed"

# Per-tool plan: list of (param_name, Selector_or_None, entity_type).
# ``Selector is None`` means "use HydrationMap fallback for entity_type".
_ToolPlan = list[tuple[str, Selector | None, type]]

# pydantic-ai's ToolPrepareFunc signature allows either a sync or async
# return; we always install async ourselves but also accept sync chained
# originals (we await on isawaitable).
PrepareFunc = Callable[[Any, ToolDefinition], Any]


def register_grounded_tools(
    agent: Agent[Any, Any],
    *,
    selectors: SelectorRegistry | None = None,
    hydration: HydrationMap | None = None,
) -> None:
    """Scan ``agent``'s registered tools and install grounded ``prepare`` hooks.

    For each tool, inspect parameter annotations via
    ``get_type_hints(fn, include_extras=True)``. Any parameter whose
    annotation is ``Annotated[Ref[T], Selector(...)]`` (or bare ``Ref[T]``
    with a HydrationMap fallback available) gets its JSON Schema narrowed
    at run-time to a closed enum of valid IDs.

    Tools without any Ref-annotated parameters are left alone.
    """
    toolset = agent._function_toolset  # noqa: SLF001 — public API gap
    tools: dict[str, Tool[Any]] = toolset.tools

    for tool in tools.values():
        if getattr(tool, _GROUNDED_PREPARE_MARKER, False):
            continue  # idempotent — already installed

        plan = _plan_for_tool(tool, hydration=hydration)
        if not plan:
            continue

        original_prepare: PrepareFunc | None = tool.prepare
        new_prepare = _make_prepare(
            plan=plan,
            selectors=selectors,
            hydration=hydration,
            original=original_prepare,
        )
        tool.prepare = new_prepare
        setattr(tool, _GROUNDED_PREPARE_MARKER, True)


def _plan_for_tool(
    tool: Tool[Any],
    *,
    hydration: HydrationMap | None,
) -> _ToolPlan:
    """Return the list of (param, selector, entity_type) for grounded params."""
    fn = tool.function
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        return []

    plan: _ToolPlan = []
    for param_name, annotation in hints.items():
        if param_name == "return":
            continue
        entity_type, selector = extract_selector(annotation)
        if entity_type is None:
            continue
        if selector is None and hydration is None:
            # Bare Ref[T] without HydrationMap — nothing to ground against.
            continue
        plan.append((param_name, selector, entity_type))
    return plan


def _make_prepare(
    *,
    plan: _ToolPlan,
    selectors: SelectorRegistry | None,
    hydration: HydrationMap | None,
    original: PrepareFunc | None,
) -> PrepareFunc:
    async def prepare(ctx: Any, tool_def: ToolDefinition) -> ToolDefinition | None:
        # Chain the original prepare first (if any). If it hides the
        # tool or rewrites the schema, we respect that as the starting point.
        if original is not None:
            raw_result: Any = original(ctx, tool_def)
            if inspect.isawaitable(raw_result):
                raw_result = await raw_result
            if raw_result is None:
                return None
            tool_def = raw_result

        props = dict(tool_def.parameters_json_schema.get("properties", {}))
        for param_name, selector, entity_type in plan:
            ids = await _resolve_ids(
                selector=selector,
                entity_type=entity_type,
                ctx=ctx,
                selectors=selectors,
                hydration=hydration,
            )
            if not ids:
                return None  # hide the tool — no valid IDs to reference

            param_schema = dict(props.get(param_name, {}))
            param_schema["enum"] = [str(i) for i in ids]
            # Best-effort title-preview: works when the selector returned
            # entity instances (with a `.title` or `.name`) OR when a
            # HydrationMap exposes `preview_for(ids)`. For pure UUID lists
            # we just list the IDs.
            existing_desc = param_schema.get(
                "description",
                f"ID of an existing {entity_type.__name__}",
            )
            param_schema["description"] = (
                f"{existing_desc} — MUST be one of {len(ids)} "
                f"valid {entity_type.__name__} id(s)."
            )
            props[param_name] = param_schema

        new_schema = dict(tool_def.parameters_json_schema)
        new_schema["properties"] = props
        return dataclasses_replace(tool_def, parameters_json_schema=new_schema)

    return prepare


async def _resolve_ids(
    *,
    selector: Selector | None,
    entity_type: type,
    ctx: Any,
    selectors: SelectorRegistry | None,
    hydration: HydrationMap | None,
) -> list[Any]:
    """Resolve to a list of UUIDs for one parameter."""
    if selector is not None:
        return await resolve_selector_ids(selector, ctx, selectors)
    # HydrationMap fallback — for a bare Ref[T] we want the repo's
    # default "list everything for tenant" behavior. HydrationMap doesn't
    # currently expose a `list_all` method, so this branch is reachable
    # only if a future HydrationMap grows one. For now: no Selector +
    # no extension → return [] (which hides the tool, signaling
    # misconfiguration).
    if hydration is not None:
        repo = getattr(hydration, "_repos", {}).get(entity_type)
        if repo is not None and hasattr(repo, "list_all"):
            return list(await repo.list_all(ctx))
    return []

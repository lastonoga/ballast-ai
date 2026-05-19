"""Selector — per-use-site closed-set picker for ``Ref[T]``.

A ``Selector`` is attached as ``Annotated`` metadata to a ``Ref[T]`` field
(on an output BaseModel) or to a tool parameter. Downstream scanners
(``GroundedAgent`` for output fields, ``register_grounded_tools`` for tool
inputs) read the metadata and call the selector at run-time to determine
the closed set of allowed UUIDs.

Two styles:

- **inline**: ``Selector(lambda ctx: ctx.deps.repo.list_open(...))``
- **named**:  ``Selector("open_notes")`` — resolved via a
  ``SelectorRegistry`` shared across many tools/output fields.

Without a ``Selector``, callers fall back to the legacy ``HydrationMap``
"list everything for tenant" behavior.

Schema-level note: ``Ref[T]`` itself does NOT change shape. The
selector lives purely in ``Annotated`` metadata, which Pydantic ignores
during schema generation. ``get_type_hints(..., include_extras=True)``
is how the scanners pull it back out.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypeVar, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded.ref import Ref

T = TypeVar("T")

# Async (or sync) function that returns the closed-set of allowed
# UUIDs (or entity instances) for this use-site.
#
# We accept ``Any`` for the context to avoid framework-side coupling to
# pydantic-ai's ``RunContext`` type at module load. The scanners pass the
# real ``RunContext`` through unchanged.
SelectorFunc = Callable[[Any], Awaitable[list[Any]] | list[Any]]


@dataclass(frozen=True)
class Selector:
    """``Annotated[Ref[T], Selector(...)]`` picks the closed-set per use-site.

    The single field ``fn_or_name`` holds either a callable (inline) or
    a string registry key (named). ``resolve(registry)`` returns the
    actual callable, looking up named selectors against the registry.
    """

    fn_or_name: SelectorFunc | str

    def is_named(self) -> bool:
        return isinstance(self.fn_or_name, str)

    def resolve(self, registry: SelectorRegistry | None) -> SelectorFunc:
        if isinstance(self.fn_or_name, str):
            if registry is None:
                raise KeyError(
                    f"Selector({self.fn_or_name!r}) is named but no "
                    "SelectorRegistry was provided to the scanner.",
                )
            return registry.get(self.fn_or_name)
        return self.fn_or_name


class SelectorRegistry:
    """Named-selector storage.

    Apps construct one and pass it to ``GroundedAgent`` /
    ``register_grounded_tools``. Names are flat (no namespacing); apps
    own the naming convention.
    """

    def __init__(self) -> None:
        self._fns: dict[str, SelectorFunc] = {}

    def register(self, name: str, fn: SelectorFunc) -> None:
        if name in self._fns:
            raise ValueError(f"SelectorRegistry: {name!r} already registered")
        self._fns[name] = fn

    def get(self, name: str) -> SelectorFunc:
        try:
            return self._fns[name]
        except KeyError as exc:
            available = sorted(self._fns)
            raise KeyError(
                f"SelectorRegistry: no selector named {name!r}. "
                f"Registered: {available}",
            ) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._fns


def extract_selector(annotated_type: Any) -> tuple[type[BaseModel] | None, Selector | None]:
    """Walk ``Annotated[Ref[T], Selector(...), ...]``.

    Returns ``(entity_type, selector)``:

    - ``(EntityType, Selector(...))`` for ``Annotated[Ref[T], Selector(...)]``
    - ``(EntityType, None)`` for bare ``Ref[T]`` or ``Annotated[Ref[T], ...]``
      with no Selector
    - ``(None, None)`` if the annotation is not a ``Ref`` at all

    Recurses through ``Optional[...]`` and ``list[...]`` wrappers so the
    same helper works for output-field and tool-param introspection.
    """
    origin = get_origin(annotated_type)
    args = get_args(annotated_type)

    # Annotated[X, meta1, meta2, ...] — first arg is the inner type, rest are metas.
    if origin is Annotated or (origin is not None and hasattr(annotated_type, "__metadata__")):
        inner = args[0]
        # Metadata may be on annotated_type.__metadata__ for typing.Annotated.
        metadata = getattr(annotated_type, "__metadata__", args[1:])
        selector = next((m for m in metadata if isinstance(m, Selector)), None)
        target, _inner_selector = extract_selector(inner)
        # Outer Selector wins over inner (Annotated should not normally nest).
        return target, selector or _inner_selector

    # Bare Ref[T]
    if isinstance(annotated_type, type) and issubclass(annotated_type, Ref) and annotated_type is not Ref:
        target = getattr(annotated_type, "__entity_type__", None)
        return target, None

    return None, None


async def resolve_selector_ids(
    selector: Selector,
    ctx: Any,
    registry: SelectorRegistry | None,
) -> list[UUID]:
    """Call the selector with ``ctx`` and normalize the result to a list of UUIDs.

    Handles:

    - sync vs async callables (``inspect.isawaitable`` on the return)
    - list[UUID]   → returned as-is
    - list[entity] → extracts ``.id`` from each
    - empty list   → returned as ``[]`` (callers decide what to do)
    """
    fn = selector.resolve(registry)
    result = fn(ctx)
    if inspect.isawaitable(result):
        result = await result
    out: list[UUID] = []
    for item in result:
        if isinstance(item, UUID):
            out.append(item)
        elif isinstance(item, Ref):
            out.append(item.id)
        else:
            # Entity instance — extract `.id`
            id_val = getattr(item, "id", None)
            if id_val is None:
                raise TypeError(
                    f"Selector returned {type(item).__name__} without an .id "
                    "attribute; expected UUID, Ref, or entity with .id",
                )
            out.append(id_val if isinstance(id_val, UUID) else UUID(str(id_val)))
    return out

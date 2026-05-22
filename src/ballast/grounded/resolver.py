from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from ballast.grounded._build import build_dynamic
from ballast.grounded._scan import scan_context, scan_output
from ballast.grounded._spec import ContextSources, FieldRole, FieldSpec, OutputSpec
from ballast.grounded.errors import GroundedBuildError
from ballast.grounded.selector import (
    SelectorRegistry,
    resolve_selector_ids,
)


class GroundedResolver:
    """Per-Pattern scanner + dynamic-model builder.

    The output type's OutputSpec is cached at construction (it's static
    per Pattern), while `build` is called per-run with a context that
    varies. Returns (DynamicModel, OutputSpec) so callers like
    HydrationMap (Task 18) can re-traverse the structure.

    Two entry points:

    - ``build(context, constraints=...)`` — sync, backward-compatible.
      Uses ``HydrationMap``-style context-scanning to collect IDs.
    - ``abuild(context, ...)`` — async, honors ``Selector`` metadata on
      ``Annotated[Ref[T], Selector(...)]`` fields. Selectors take
      precedence over context-scan IDs.
    """

    def __init__(self, output_type: type[BaseModel]) -> None:
        self.output_type = output_type
        self._spec: OutputSpec = scan_output(output_type)

    def build(
        self,
        context: BaseModel,
        constraints: dict[str, Any] | None = None,
    ) -> tuple[type[BaseModel], OutputSpec]:
        sources = scan_context(context, self._spec)
        if constraints:
            sources = self._apply_constraints(sources, constraints)
        dynamic = build_dynamic(self.output_type, self._spec, sources)
        return dynamic, self._spec

    async def abuild(
        self,
        context: BaseModel,
        *,
        selector_ctx: Any | None = None,
        selectors: SelectorRegistry | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> tuple[type[BaseModel], OutputSpec]:
        """Async build that resolves any ``Selector`` metadata on output fields.

        ``selector_ctx`` is passed to each selector function (typically a
        ``RunContext`` or a deps-bearing object the lambda closes over).
        Falls back to context-scan for fields with no Selector, preserving
        backward compatibility with HydrationMap-style grounding.
        """
        sources = scan_context(context, self._spec)
        # Layer Selector IDs on top — they win over context-scan results
        # for the specific entity types they target.
        for fspec in _ref_fields(self._spec):
            if fspec.selector is None:
                continue
            if fspec.target_type is None:
                continue
            ids = await resolve_selector_ids(fspec.selector, selector_ctx, selectors)
            sources.by_entity_type[fspec.target_type] = list(ids)
        if constraints:
            sources = self._apply_constraints(sources, constraints)
        dynamic = build_dynamic(self.output_type, self._spec, sources)
        return dynamic, self._spec

    def _apply_constraints(
        self, sources: ContextSources, constraints: dict[str, Any]
    ) -> ContextSources:
        for path, value in constraints.items():
            fspec = self._find_field_by_path(path)
            if fspec is None:
                raise GroundedBuildError(
                    f"constraints[{path!r}]: unknown path in output type"
                )
            if fspec.role not in (FieldRole.REF, FieldRole.LIST_REF, FieldRole.OPTIONAL_REF):
                raise GroundedBuildError(
                    f"constraints[{path!r}]: path role {fspec.role.value} not "
                    "overridable in v1 (only REF / LIST_REF / OPTIONAL_REF)"
                )
            values = value if isinstance(value, list) else [value]
            # Coerce strings to UUID for convenience
            coerced = [UUID(v) if isinstance(v, str) else v for v in values]
            if fspec.target_type is not None:
                sources.by_entity_type[fspec.target_type] = coerced
        return sources

    def _find_field_by_path(self, path: str) -> FieldSpec | None:
        # v1: only top-level paths (no dotted nesting / no [*] glob).
        # Nested-path support deferred — would require resolver walking
        # nested specs to locate the target field.
        if "." in path or "[" in path:
            raise GroundedBuildError(
                f"constraints[{path!r}]: nested / list paths not supported in v1 "
                "(only top-level field names)"
            )
        return self._spec.fields.get(path)


def _ref_fields(spec: OutputSpec) -> list[FieldSpec]:
    """Recursively yield all REF-shaped fields under ``spec`` (incl. nested)."""
    out: list[FieldSpec] = []
    for f in spec.fields.values():
        if f.role in (FieldRole.REF, FieldRole.LIST_REF, FieldRole.OPTIONAL_REF):
            out.append(f)
        elif f.role in (FieldRole.NESTED, FieldRole.LIST_NESTED) and f.nested_spec:
            out.extend(_ref_fields(f.nested_spec))
    return out

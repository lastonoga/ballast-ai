from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output
from pydantic_ai_stateflow.grounded._spec import ContextSources, FieldRole, FieldSpec, OutputSpec
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class GroundedResolver:
    """Per-Pattern scanner + dynamic-model builder.

    The output type's OutputSpec is cached at construction (it's static
    per Pattern), while `build` is called per-run with a context that
    varies. Returns (DynamicModel, OutputSpec) so callers like
    HydrationMap (Task 18) can re-traverse the structure.
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

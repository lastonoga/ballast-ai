from __future__ import annotations

import typing
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, create_model

from pydantic_ai_stateflow.grounded._spec import ContextSources, FieldRole, OutputSpec
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


def _make_literal(values: list[Any]) -> Any:
    """Create a Literal type from a list of values at runtime.

    Uses the internal ``typing._GenericAlias`` to construct the Literal because
    the public API requires static arguments. This is stable across Python 3.11+
    and is the accepted pattern for dynamic Literal construction.
    """
    return typing._GenericAlias(Literal, tuple(values))  # type: ignore[attr-defined]


def build_dynamic(
    model: type[BaseModel],
    spec: OutputSpec,
    sources: ContextSources,
) -> type[BaseModel]:
    """Build a dynamic Pydantic model where Ref/Enum fields become Literals.

    Recursive (in future tasks): nested BaseModel fields will be rebuilt with
    dynamic Literals. Existing (non-grounded) fields are passed through.
    """
    fields: dict[str, Any] = {}
    for name, fspec in spec.fields.items():
        field_info = model.model_fields[name]
        match fspec.role:
            case FieldRole.REF:
                target = fspec.target_type
                assert target is not None, "REF field must have a target_type"
                ids = sources.by_entity_type.get(target, [])
                if not ids:
                    raise GroundedBuildError(
                        f"No instances of {target.__name__} in context "
                        f"for {fspec.path}"
                    )
                # Pydantic stores the inner Literal as `annotation` on FieldInfo,
                # so `get_origin(field.annotation) is Literal` holds.
                # `BeforeValidator(UUID)` coerces string input to UUID before the
                # Literal membership check, so LLM string output validates cleanly.
                literal_type = _make_literal(ids)
                annotation = Annotated[literal_type, BeforeValidator(UUID)]  # type: ignore[valid-type]
                fields[name] = (annotation, field_info)

            case FieldRole.FREE:
                fields[name] = (field_info.annotation, field_info)

            case _:
                # Other roles handled in subsequent tasks; passthrough for now
                fields[name] = (field_info.annotation, field_info)

    return create_model(f"Dynamic{model.__name__}", __base__=BaseModel, **fields)

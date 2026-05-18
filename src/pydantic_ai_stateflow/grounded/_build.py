from __future__ import annotations

import typing
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, BeforeValidator, create_model

from pydantic_ai_stateflow.grounded._spec import ContextSources, FieldRole, OutputSpec
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError
from pydantic_ai_stateflow.grounded.ref import Ref


def _make_literal(values: tuple[Any, ...]) -> Any:
    """Create a Literal type from values at runtime.

    Public Literal[...] subscription needs static args at parse time, so we
    go through typing._GenericAlias for dynamic construction. Stable across
    Python 3.11+; documented private-ish entry point.
    """
    return typing._GenericAlias(Literal, values)  # type: ignore[attr-defined]


def _make_ref_wrapper(entity_type: type[BaseModel]) -> Any:
    """Build an AfterValidator that wraps a validated UUID into Ref[entity_type].

    This is what makes `result.value.chosen` come back as a `Ref[Entity]` instance
    (matching the user's static type) rather than a bare UUID.
    """
    def _wrap(value: Any) -> Ref[Any]:
        if isinstance(value, Ref):
            # Re-wrap to the exact subscripted class so entity_type is correct.
            return Ref[entity_type](value.id)  # type: ignore[valid-type]
        if isinstance(value, UUID):
            return Ref[entity_type](value)  # type: ignore[valid-type]
        # Stringified UUID — coerce.
        return Ref[entity_type](UUID(value))  # type: ignore[valid-type]
    return AfterValidator(_wrap)


def build_dynamic(
    model: type[BaseModel],
    spec: OutputSpec,
    sources: ContextSources,
) -> type[BaseModel]:
    """Build a dynamic Pydantic model where Ref fields become Literal[*ids]
    + an AfterValidator that wraps the matched UUID back into Ref[Entity].

    User code sees `result.value.chosen` as `Ref[Entity]` at runtime, matching
    the static type signature. The Literal layer is what gives the LLM-facing
    JSON Schema its `enum` constraint.

    Recursive (in future tasks): nested BaseModel fields will be rebuilt with
    dynamic Literals. Other (non-grounded) fields are passed through.
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
                # Literal of stringified UUIDs → JSON schema enum advertises
                # allowed values to the LLM as strings.
                # BeforeValidator: coerce input UUID/Ref to string for the
                # Literal membership check.
                # AfterValidator: wrap matched string into Ref[Entity] so
                # downstream code receives the typed reference.
                id_strs = tuple(str(i) for i in ids)
                literal_type: Any = _make_literal(id_strs)
                _before = BeforeValidator(
                    lambda v: str(v.id) if isinstance(v, Ref)
                    else str(v) if isinstance(v, UUID)
                    else v
                )
                annotation = Annotated[literal_type, _before, _make_ref_wrapper(target)]
                fields[name] = (annotation, field_info)

            case FieldRole.FREE:
                fields[name] = (field_info.annotation, field_info)

            case _:
                # Other roles handled in subsequent tasks; passthrough for now
                fields[name] = (field_info.annotation, field_info)

    return create_model(f"Dynamic{model.__name__}", __base__=BaseModel, **fields)

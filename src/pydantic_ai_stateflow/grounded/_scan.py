from __future__ import annotations

from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded._spec import FieldRole, FieldSpec, OutputSpec
from pydantic_ai_stateflow.grounded.ref import Ref


def scan_output(model: type[BaseModel], path: str = "") -> OutputSpec:
    """Walk Pydantic model fields and classify each by its role.

    Recurses into nested BaseModel and list[BaseModel] fields. Stops at
    primitive / unrecognised fields (FieldRole.FREE).
    """
    spec = OutputSpec(model=model)
    for name, info in model.model_fields.items():
        full_path = f"{path}.{name}" if path else name
        spec.fields[name] = _classify(name, full_path, info.annotation)
    return spec


def _classify(name: str, path: str, annotation: Any) -> FieldSpec:
    # Direct Ref[X]
    if _is_ref_type(annotation):
        return FieldSpec(name=name, path=path, role=FieldRole.REF, target_type=_ref_target(annotation))

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[Ref[X]] / Union[Ref[X], None]
    if origin in (Union, UnionType):
        non_none = [a for a in args if a is not NoneType and a is not type(None)]
        if len(non_none) == 1 and _is_ref_type(non_none[0]):
            return FieldSpec(
                name=name, path=path, role=FieldRole.OPTIONAL_REF, target_type=_ref_target(non_none[0])
            )

    # list[Ref[X]] / list[BaseModel] / list[primitive]
    if origin is list and args:
        inner = args[0]
        if _is_ref_type(inner):
            return FieldSpec(name=name, path=path, role=FieldRole.LIST_REF, target_type=_ref_target(inner))
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return FieldSpec(
                name=name, path=path, role=FieldRole.LIST_NESTED,
                target_type=inner, nested_spec=scan_output(inner, path=f"{path}[*]"),
            )

    # Literal[...]
    if origin is Literal:
        return FieldSpec(name=name, path=path, role=FieldRole.LITERAL, literal_values=args)

    # Nested BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return FieldSpec(
            name=name, path=path, role=FieldRole.NESTED,
            target_type=annotation, nested_spec=scan_output(annotation, path=path),
        )

    # Free (primitive / unrecognised — leave as-is)
    return FieldSpec(name=name, path=path, role=FieldRole.FREE)


def _is_ref_type(annotation: Any) -> bool:
    """True iff annotation is `Ref[SomeEntity]` (subscripted form)."""
    return isinstance(annotation, type) and issubclass(annotation, Ref) and annotation is not Ref


def _ref_target(annotation: Any) -> type[BaseModel]:
    target: type[BaseModel] | None = getattr(annotation, "__entity_type__", None)
    if target is None:
        raise TypeError(f"Subscripted Ref expected, got {annotation!r}")
    return target

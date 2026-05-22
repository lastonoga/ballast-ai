from __future__ import annotations

import warnings
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from ballast.grounded._spec import ContextSources, FieldRole, FieldSpec, OutputSpec
from ballast.grounded.ref import Ref
from ballast.grounded.selector import Selector


def scan_output(model: type[BaseModel], path: str = "") -> OutputSpec:
    """Walk Pydantic model fields and classify each by its role.

    Recurses into nested BaseModel and list[BaseModel] fields. Stops at
    primitive / unrecognised fields (FieldRole.FREE).

    Reads ``Annotated[Ref[T], Selector(...)]`` metadata via Pydantic's
    own ``FieldInfo.metadata`` (which preserves all Annotated extras),
    avoiding ``get_type_hints`` pitfalls with forward refs / classes
    defined inside functions.
    """
    spec = OutputSpec(model=model)
    for name, info in model.model_fields.items():
        full_path = f"{path}.{name}" if path else name
        selector = next(
            (m for m in info.metadata if isinstance(m, Selector)),
            None,
        )
        spec.fields[name] = _classify(name, full_path, info.annotation, selector)
    return spec


def _classify(name: str, path: str, annotation: Any, selector: Selector | None = None) -> FieldSpec:
    # Direct Ref[X]
    if _is_ref_type(annotation):
        return FieldSpec(
            name=name, path=path, role=FieldRole.REF,
            target_type=_ref_target(annotation), selector=selector,
        )

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[Ref[X]] / Union[Ref[X], None]
    if origin in (Union, UnionType):
        non_none = [a for a in args if a is not NoneType and a is not type(None)]
        if len(non_none) == 1 and _is_ref_type(non_none[0]):
            return FieldSpec(
                name=name, path=path, role=FieldRole.OPTIONAL_REF,
                target_type=_ref_target(non_none[0]), selector=selector,
            )

    # list[Ref[X]] / list[BaseModel] / list[primitive]
    if origin is list and args:
        inner = args[0]
        if _is_ref_type(inner):
            return FieldSpec(
                name=name, path=path, role=FieldRole.LIST_REF,
                target_type=_ref_target(inner), selector=selector,
            )
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

    # Free (primitive / unrecognised — leave as-is).
    #
    # KNOWN LIMITATIONS (planned for expanded type support in v1.1):
    # - `Optional[list[Ref[X]]]`    → FREE (should be optional LIST_REF)
    # - `Optional[BaseModel]`        → FREE (should be optional NESTED)
    # - `Union[Ref[A], Ref[B]]`      → FREE (multi-type ref; ambiguous)
    # - `dict[...]`                  → FREE (intentional; not in scope)
    #
    # Silent FREE means Refs inside these constructs do NOT get Literal
    # narrowing, so the LLM could still hallucinate UUIDs there. We emit
    # a warning to surface this latent gap.
    if _contains_ref(annotation):
        warnings.warn(
            f"Field {path!r} has a Ref inside an unsupported type construct "
            f"({annotation!r}). Classified as FREE — no Literal narrowing. "
            "Use simpler shapes (bare Ref, list[Ref], Optional[Ref]) or wait "
            "for v1.1 expanded support.",
            stacklevel=3,
        )
    return FieldSpec(name=name, path=path, role=FieldRole.FREE)


def _contains_ref(annotation: Any) -> bool:
    """True iff annotation is or contains a subscripted Ref anywhere in its
    type tree. Used to warn about silent FREE-fallback on unhandled constructs."""
    if _is_ref_type(annotation):
        return True
    return any(_contains_ref(arg) for arg in get_args(annotation))


def _is_ref_type(annotation: Any) -> bool:
    """True iff annotation is `Ref[SomeEntity]` (subscripted form)."""
    return isinstance(annotation, type) and issubclass(annotation, Ref) and annotation is not Ref


def _ref_target(annotation: Any) -> type[BaseModel]:
    target: type[BaseModel] | None = getattr(annotation, "__entity_type__", None)
    if target is None:
        raise TypeError(f"Subscripted Ref expected, got {annotation!r}")
    return target


def scan_context(context: BaseModel, output_spec: OutputSpec, *, max_depth: int = 5) -> ContextSources:
    """Walk a Pydantic context, collect instances of types referenced by output_spec.

    Returns a `ContextSources` with `by_entity_type[T]` mapping to all `t.id`
    values for each `T` instance encountered, recursively. Also collects
    Literal-typed field values for enum intersection (independent of entity targets).
    """
    sources = ContextSources()
    targets = output_spec.referenced_entity_types
    _walk(context, targets, sources, depth=0, max_depth=max_depth)
    return sources


def _walk(obj: Any, targets: set[type], sources: ContextSources, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        return
    if isinstance(obj, BaseModel):
        if type(obj) in targets:
            id_val = getattr(obj, "id", None)
            if id_val is not None:
                sources.by_entity_type.setdefault(type(obj), []).append(id_val)
        for field_name, info in type(obj).model_fields.items():
            value = getattr(obj, field_name)
            # Capture Literal-typed field values for enum intersection
            if get_origin(info.annotation) is Literal:
                key = ContextSources.literal_key(get_args(info.annotation))
                sources.by_literal_values.setdefault(key, set()).add(value)
            _walk(value, targets, sources, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _walk(item, targets, sources, depth + 1, max_depth)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, targets, sources, depth + 1, max_depth)
    # primitives — stop

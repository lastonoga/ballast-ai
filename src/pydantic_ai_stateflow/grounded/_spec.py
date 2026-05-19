from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FieldRole(StrEnum):
    REF = "ref"
    LIST_REF = "list_ref"
    OPTIONAL_REF = "optional_ref"
    NESTED = "nested"
    LIST_NESTED = "list_nested"
    LITERAL = "literal"
    FREE = "free"


@dataclass
class FieldSpec:
    """One field's role within an output template."""

    name: str
    path: str
    role: FieldRole
    target_type: type | None = None
    literal_values: tuple[Any, ...] | None = None
    nested_spec: OutputSpec | None = None
    # Per-field Selector pulled from ``Annotated[Ref[T], Selector(...)]``.
    # When present, the resolver should call this Selector instead of
    # pulling IDs from ``ContextSources``. ``Any`` to avoid circular import
    # with grounded.selector; the resolver casts on use.
    selector: Any | None = None


@dataclass
class OutputSpec:
    """All fields of a single Pydantic model with their roles."""

    model: type
    fields: dict[str, FieldSpec] = field(default_factory=dict)

    @property
    def referenced_entity_types(self) -> set[type]:
        out: set[type] = set()
        for f in self.fields.values():
            if f.role in (FieldRole.REF, FieldRole.LIST_REF, FieldRole.OPTIONAL_REF):
                if f.target_type is not None:
                    out.add(f.target_type)
            elif f.role in (FieldRole.NESTED, FieldRole.LIST_NESTED) and f.nested_spec:
                out |= f.nested_spec.referenced_entity_types
        return out


@dataclass
class ContextSources:
    by_entity_type: dict[type, list[Any]] = field(default_factory=dict)
    # Track observed Literal-field values keyed by a stable string fingerprint
    # of the Literal's allowed values (e.g. "draft|ready|sent" — sorted) so
    # different Literal types with the same value set share intersections.
    by_literal_values: dict[str, set[Any]] = field(default_factory=dict)

    @staticmethod
    def literal_key(args: tuple[Any, ...]) -> str:
        return "|".join(sorted(str(a) for a in args))

"""scan_output recurses into Scored.value finding Ref[T] fields naturally.

scan_output is a *schema-level* walker: it takes a class (type[BaseModel])
and returns an OutputSpec. Scored[_Note] produces a concrete Pydantic class
whose ``value`` field is typed as _Note — a nested BaseModel — so scan_output
classifies it as NESTED and recurses, discovering Ref[_Project] without any
special-case wiring.

No source changes are expected or made here.
"""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from ballast.grounded import Ref
from ballast.grounded._scan import scan_output
from ballast.grounded._spec import FieldRole
from ballast.quality.scored._model import Scored


class _Project(BaseModel):
    id: UUID
    name: str


class _Note(BaseModel):
    title: str
    project: Ref[_Project]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scan_output_finds_ref_inside_scored_value() -> None:
    """Scored[_Note].value is NESTED → nested_spec exposes Ref[_Project]."""
    spec = scan_output(Scored[_Note])

    value_field = spec.fields["value"]
    assert value_field.role == FieldRole.NESTED, (
        f"Expected value field to be NESTED, got {value_field.role!r}"
    )
    assert value_field.nested_spec is not None
    assert value_field.target_type is _Note

    project_field = value_field.nested_spec.fields["project"]
    assert project_field.role == FieldRole.REF, (
        f"Expected project field to be REF, got {project_field.role!r}"
    )
    assert project_field.target_type is _Project


def test_scan_output_finds_refs_inside_scored_list_value() -> None:
    """Scored[list[_Note]].value is LIST_NESTED → nested_spec exposes Ref[_Project]."""
    spec = scan_output(Scored[list[_Note]])

    value_field = spec.fields["value"]
    assert value_field.role == FieldRole.LIST_NESTED, (
        f"Expected value field to be LIST_NESTED, got {value_field.role!r}"
    )
    assert value_field.nested_spec is not None
    assert value_field.target_type is _Note

    project_field = value_field.nested_spec.fields["project"]
    assert project_field.role == FieldRole.REF
    assert project_field.target_type is _Project


def test_scan_output_ignores_rationale_and_confidence_fields() -> None:
    """rationale stays FREE, confidence stays LITERAL — no phantom refs."""
    spec = scan_output(Scored[_Note])

    rationale_field = spec.fields["rationale"]
    assert rationale_field.role == FieldRole.FREE, (
        f"rationale should be FREE (plain str), got {rationale_field.role!r}"
    )

    confidence_field = spec.fields["confidence"]
    assert confidence_field.role == FieldRole.LITERAL, (
        f"confidence should be LITERAL, got {confidence_field.role!r}"
    )


def test_scan_output_referenced_entity_types() -> None:
    """OutputSpec.referenced_entity_types finds _Project through NESTED value."""
    spec = scan_output(Scored[_Note])
    entity_types = spec.referenced_entity_types
    assert _Project in entity_types, (
        f"_Project not found in referenced_entity_types: {entity_types!r}"
    )

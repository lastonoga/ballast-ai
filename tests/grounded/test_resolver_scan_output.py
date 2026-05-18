from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._scan import scan_output
from pydantic_ai_stateflow.grounded._spec import FieldRole


class Item(BaseModel):
    id: UUID
    name: str


class Status(BaseModel):
    state: Literal["draft", "ready", "sent"]


def test_scan_detects_ref_field():
    class Out(BaseModel):
        chosen: Ref[Item]
        rationale: str

    spec = scan_output(Out)
    assert spec.fields["chosen"].role == FieldRole.REF
    assert spec.fields["chosen"].target_type is Item
    assert spec.fields["rationale"].role == FieldRole.FREE


def test_scan_detects_list_of_refs():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    spec = scan_output(Out)
    assert spec.fields["chosen"].role == FieldRole.LIST_REF
    assert spec.fields["chosen"].target_type is Item


def test_scan_detects_optional_ref():
    class Out(BaseModel):
        maybe: Optional[Ref[Item]]  # noqa: UP007,UP045 — explicit Optional for test

    spec = scan_output(Out)
    assert spec.fields["maybe"].role == FieldRole.OPTIONAL_REF
    assert spec.fields["maybe"].target_type is Item


def test_scan_detects_nested_model():
    class Inner(BaseModel):
        chosen: Ref[Item]

    class Out(BaseModel):
        inner: Inner

    spec = scan_output(Out)
    assert spec.fields["inner"].role == FieldRole.NESTED
    assert spec.fields["inner"].nested_spec is not None
    assert spec.fields["inner"].nested_spec.fields["chosen"].role == FieldRole.REF


def test_scan_detects_list_of_nested_models():
    class Inner(BaseModel):
        chosen: Ref[Item]

    class Out(BaseModel):
        items: list[Inner]

    spec = scan_output(Out)
    assert spec.fields["items"].role == FieldRole.LIST_NESTED
    assert spec.fields["items"].nested_spec is not None


def test_scan_detects_literal_field():
    class Out(BaseModel):
        state: Literal["draft", "ready", "sent"]

    spec = scan_output(Out)
    assert spec.fields["state"].role == FieldRole.LITERAL
    assert spec.fields["state"].literal_values == ("draft", "ready", "sent")

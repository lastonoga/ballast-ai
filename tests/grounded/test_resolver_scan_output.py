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


def test_scan_warns_when_ref_buried_in_unsupported_construct(recwarn):
    """Known v1 limitation: Optional[list[Ref[X]]] / Union[Ref[A], Ref[B]] /
    Optional[BaseModel-with-Ref] silently classify as FREE. Emit warning so
    developers see the latent gap instead of hallucination risk hiding."""
    import warnings

    class Out(BaseModel):
        maybe_list: Optional[list[Ref[Item]]] = None  # noqa: UP007,UP045

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        spec = scan_output(Out)

    assert spec.fields["maybe_list"].role == FieldRole.FREE
    matched = [w for w in recwarn.list if "FREE" in str(w.message) and "maybe_list" in str(w.message)]
    assert len(matched) >= 1, f"Expected FREE-fallback warning, got: {[str(w.message) for w in recwarn.list]}"

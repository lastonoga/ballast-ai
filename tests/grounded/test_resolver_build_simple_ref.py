from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class Item(BaseModel):
    id: UUID
    name: str


def test_build_advertises_ids_as_json_schema_enum_for_llm():
    """Dynamic model's REF field must expose allowed ids as JSON Schema
    `enum` of stringified UUIDs so the LLM sees the closed set."""
    class Out(BaseModel):
        chosen: Ref[Item]
        rationale: str

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    schema = Dynamic.model_json_schema()
    chosen_schema = schema["properties"]["chosen"]
    assert "enum" in chosen_schema
    assert set(chosen_schema["enum"]) == {str(i) for i in ids}


def test_build_validation_returns_typed_ref_for_allowed_value():
    """User-visible runtime: obj.chosen must be a Ref[Item] (matching the
    static type), not a bare UUID or string. This is what makes the
    downstream `result.hydrate(repo)` API work."""
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    obj = Dynamic.model_validate({"chosen": str(ids[0])})
    assert isinstance(obj.chosen, Ref)
    assert obj.chosen.id == ids[0]
    assert obj.chosen.entity_type is Item


def test_build_validation_rejects_unknown_value():
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": str(uuid4())})


def test_build_raises_when_no_entities_in_context():
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        unrelated: str

    ctx = Ctx(unrelated="x")
    with pytest.raises(GroundedBuildError, match="No instances of Item"):
        build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

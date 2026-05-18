from typing import Literal, get_args, get_origin
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


def test_build_replaces_ref_with_literal_of_ids():
    class Out(BaseModel):
        chosen: Ref[Item]
        rationale: str

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    out_spec = scan_output(Out)
    sources = scan_context(ctx, out_spec)
    Dynamic = build_dynamic(Out, out_spec, sources)  # noqa: N806

    chosen_field = Dynamic.model_fields["chosen"]
    assert get_origin(chosen_field.annotation) is Literal
    assert set(get_args(chosen_field.annotation)) == set(ids)


def test_build_validation_passes_for_allowed_value():
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    obj = Dynamic.model_validate({"chosen": str(ids[0])})
    assert obj.chosen == ids[0]


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

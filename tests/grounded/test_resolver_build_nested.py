from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Item(BaseModel):
    id: UUID
    name: str


def test_nested_basemodel_with_ref_field_recursively_built():
    class Inner(BaseModel):
        chosen: Ref[Item]
        rationale: str

    class Out(BaseModel):
        inner: Inner

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    # Valid id round-trip
    obj = Dynamic.model_validate({"inner": {"chosen": str(ids[0]), "rationale": "r"}})
    assert isinstance(obj.inner.chosen, Ref)
    assert obj.inner.chosen.id == ids[0]
    # Invalid id rejected
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"inner": {"chosen": str(uuid4()), "rationale": "r"}})


def test_list_of_nested_models_recurses_and_broadcasts():
    class Inner(BaseModel):
        chosen: Ref[Item]
        score: int

    class Out(BaseModel):
        items: list[Inner]

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    obj = Dynamic.model_validate({"items": [
        {"chosen": str(ids[0]), "score": 1},
        {"chosen": str(ids[1]), "score": 2},
    ]})
    assert obj.items[0].chosen.id == ids[0]
    assert obj.items[1].chosen.id == ids[1]
    assert all(isinstance(it.chosen, Ref) for it in obj.items)


def test_deeply_nested_refs_all_validate():
    class Inner(BaseModel):
        chosen: Ref[Item]

    class Mid(BaseModel):
        ins: list[Inner]

    class Out(BaseModel):
        mid: Mid

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    obj = Dynamic.model_validate({"mid": {"ins": [{"chosen": str(ids[0])}]}})
    assert obj.mid.ins[0].chosen.id == ids[0]
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"mid": {"ins": [{"chosen": str(uuid4())}]}})

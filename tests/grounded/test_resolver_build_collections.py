from typing import Optional
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Item(BaseModel):
    id: UUID
    name: str


def _build(out_cls: type[BaseModel], ctx_items: list[Item]) -> type[BaseModel]:
    class Ctx(BaseModel):
        items: list[Item]

    ctx = Ctx(items=ctx_items)
    return build_dynamic(out_cls, scan_output(out_cls), scan_context(ctx, scan_output(out_cls)))


def test_list_ref_json_schema_advertises_enum():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    ids = [uuid4(), uuid4(), uuid4()]
    items = [Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)]
    Dynamic = _build(Out, items)  # noqa: N806

    schema = Dynamic.model_json_schema()
    chosen_schema = schema["properties"]["chosen"]
    assert chosen_schema["type"] == "array"
    item_schema = chosen_schema["items"]
    assert "enum" in item_schema
    assert set(item_schema["enum"]) == {str(i) for i in ids}


def test_list_ref_validation_passes_for_subset():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    ids = [uuid4(), uuid4(), uuid4()]
    items = [Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)]
    Dynamic = _build(Out, items)  # noqa: N806

    obj = Dynamic.model_validate({"chosen": [str(ids[0]), str(ids[2])]})
    assert len(obj.chosen) == 2
    assert all(isinstance(r, Ref) for r in obj.chosen)
    assert {r.id for r in obj.chosen} == {ids[0], ids[2]}


def test_list_ref_validation_rejects_unknown():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    ids = [uuid4()]
    Dynamic = _build(Out, [Item(id=ids[0], name="a")])  # noqa: N806

    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": [str(uuid4())]})


def test_optional_ref_accepts_none_and_valid_id():
    class Out(BaseModel):
        maybe: Optional[Ref[Item]] = None  # noqa: UP007,UP045

    ids = [uuid4()]
    Dynamic = _build(Out, [Item(id=ids[0], name="a")])  # noqa: N806

    # None validates
    obj_none = Dynamic.model_validate({"maybe": None})
    assert obj_none.maybe is None

    # Valid id validates and returns Ref
    obj = Dynamic.model_validate({"maybe": str(ids[0])})
    assert isinstance(obj.maybe, Ref)
    assert obj.maybe.id == ids[0]


def test_optional_ref_rejects_unknown_id():
    class Out(BaseModel):
        maybe: Optional[Ref[Item]] = None  # noqa: UP007,UP045

    ids = [uuid4()]
    Dynamic = _build(Out, [Item(id=ids[0], name="a")])  # noqa: N806

    with pytest.raises(ValidationError):
        Dynamic.model_validate({"maybe": str(uuid4())})

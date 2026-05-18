from uuid import UUID, uuid4

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Item(BaseModel):
    id: UUID
    name: str


class Customer(BaseModel):
    id: UUID
    email: str


def test_scan_collects_list_of_entities():
    class Ctx(BaseModel):
        items: list[Item]
        notes: str

    class Out(BaseModel):
        chosen: Ref[Item]

    item_ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=item_ids[0], name="a"), Item(id=item_ids[1], name="b")], notes="x")
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(item_ids)


def test_scan_collects_singleton_entity():
    class Ctx(BaseModel):
        customer: Customer

    class Out(BaseModel):
        ref: Ref[Customer]

    cust_id = uuid4()
    sources = scan_context(Ctx(customer=Customer(id=cust_id, email="x@y.z")), scan_output(Out))
    assert sources.by_entity_type[Customer] == [cust_id]


def test_scan_collects_from_nested_pydantic():
    class Holder(BaseModel):
        items: list[Item]

    class Ctx(BaseModel):
        holder: Holder

    class Out(BaseModel):
        ref: Ref[Item]

    ids = [uuid4(), uuid4(), uuid4()]
    ctx = Ctx(holder=Holder(items=[Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)]))
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(ids)


def test_scan_returns_empty_for_unreferenced_types():
    class Ctx(BaseModel):
        items: list[Item]

    class Out(BaseModel):
        unrelated: str

    sources = scan_context(Ctx(items=[Item(id=uuid4(), name="a")]), scan_output(Out))
    assert Item not in sources.by_entity_type


def test_scan_unions_multiple_sources_of_same_type():
    class Ctx(BaseModel):
        top: list[Item]
        fallback: list[Item]

    class Out(BaseModel):
        ref: Ref[Item]

    top_ids = [uuid4()]
    fb_ids = [uuid4(), uuid4()]
    ctx = Ctx(
        top=[Item(id=top_ids[0], name="t")],
        fallback=[Item(id=fb_ids[0], name="f1"), Item(id=fb_ids[1], name="f2")],
    )
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(top_ids + fb_ids)

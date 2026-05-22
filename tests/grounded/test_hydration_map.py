from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from ballast.grounded import Ref
from ballast.grounded._scan import scan_output
from ballast.grounded.hydration import HydrationMap


class Item(BaseModel):
    id: UUID
    name: str


class Decision(BaseModel):
    chosen: Ref[Item]
    rationale: str


class FakeRepo:
    def __init__(self, items: dict[UUID, Item]) -> None:
        self._items = items

    async def load(self, id: UUID) -> Item:
        return self._items[id]


@pytest.mark.asyncio
async def test_hydrate_replaces_single_ref_with_entity():
    item_id = uuid4()
    item = Item(id=item_id, name="hydrated")
    repo = FakeRepo({item_id: item})

    decision = Decision(chosen=Ref[Item](item_id), rationale="r")
    hmap = HydrationMap(scan_output(Decision))
    hydrated = await hmap.hydrate(decision, repos={Item: repo})

    assert isinstance(hydrated["chosen"], Item)
    assert hydrated["chosen"].name == "hydrated"
    assert hydrated["rationale"] == "r"


@pytest.mark.asyncio
async def test_hydrate_works_on_list_of_refs():
    class Out(BaseModel):
        items: list[Ref[Item]]

    ids = [uuid4(), uuid4()]
    items = {i: Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)}
    repo = FakeRepo(items)

    obj = Out(items=[Ref[Item](ids[0]), Ref[Item](ids[1])])
    hmap = HydrationMap(scan_output(Out))
    hydrated = await hmap.hydrate(obj, repos={Item: repo})

    assert len(hydrated["items"]) == 2
    assert all(isinstance(it, Item) for it in hydrated["items"])


@pytest.mark.asyncio
async def test_hydrate_missing_repo_raises():
    item_id = uuid4()
    decision = Decision(chosen=Ref[Item](item_id), rationale="r")
    hmap = HydrationMap(scan_output(Decision))
    with pytest.raises(KeyError, match="Item"):
        await hmap.hydrate(decision, repos={})

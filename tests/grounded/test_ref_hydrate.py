from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref


class Order(BaseModel):
    id: UUID
    amount: int


class FakeOrderRepo:
    def __init__(self, orders: dict[UUID, Order]) -> None:
        self._orders = orders

    async def load(self, id: UUID) -> Order:
        if id not in self._orders:
            raise KeyError(id)
        return self._orders[id]


@pytest.mark.asyncio
async def test_hydrate_returns_entity_from_repo():
    oid = uuid4()
    order = Order(id=oid, amount=100)
    repo = FakeOrderRepo({oid: order})

    ref = Ref[Order](oid)
    loaded = await ref.hydrate(repo)
    assert loaded is order


@pytest.mark.asyncio
async def test_hydrate_propagates_repo_errors():
    repo = FakeOrderRepo({})
    ref = Ref[Order](uuid4())
    with pytest.raises(KeyError):
        await ref.hydrate(repo)

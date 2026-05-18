from uuid import uuid4

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref


class Entity(BaseModel):
    id: str
    name: str


def test_ref_stores_id_and_entity_type():
    ent_id = uuid4()
    ref = Ref[Entity](ent_id)
    assert ref.id == ent_id
    assert ref.entity_type is Entity


def test_ref_class_getitem_creates_subscripted_class():
    subscripted = Ref[Entity]
    # Subscripted form must remember the entity type for resolver later
    assert subscripted.__entity_type__ is Entity


def test_ref_equality_by_id_and_type():
    ent_id = uuid4()
    a = Ref[Entity](ent_id)
    b = Ref[Entity](ent_id)
    assert a == b


def test_ref_inequality_different_types():
    class OtherEntity(BaseModel):
        id: str

    ent_id = uuid4()
    a = Ref[Entity](ent_id)
    b = Ref[OtherEntity](ent_id)
    assert a != b


def test_ref_subscription_is_cached():
    """Class identity: Ref[Entity] is Ref[Entity] (same class object).

    Critical for the resolver in Tasks 9-11 which compares classes by `is`.
    """
    assert Ref[Entity] is Ref[Entity]

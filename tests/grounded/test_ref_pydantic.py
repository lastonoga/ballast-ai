from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref


class Item(BaseModel):
    id: UUID
    name: str


class Decision(BaseModel):
    chosen: Ref[Item]
    rationale: str


def test_decision_serializes_ref_as_uuid_string():
    item_id = uuid4()
    d = Decision(chosen=Ref[Item](item_id), rationale="best fit")
    dumped = d.model_dump(mode="json")
    assert dumped == {"chosen": str(item_id), "rationale": "best fit"}


def test_decision_deserializes_uuid_string_to_ref():
    item_id = uuid4()
    d = Decision.model_validate({"chosen": str(item_id), "rationale": "best fit"})
    assert isinstance(d.chosen, Ref)
    assert d.chosen.id == item_id
    assert d.chosen.entity_type is Item


def test_decision_roundtrip_via_json():
    item_id = uuid4()
    original = Decision(chosen=Ref[Item](item_id), rationale="r")
    restored = Decision.model_validate_json(original.model_dump_json())
    assert restored.chosen == original.chosen
    assert restored.rationale == original.rationale


def test_decision_rejects_non_uuid_string():
    with pytest.raises(ValidationError):
        Decision.model_validate({"chosen": "not-a-uuid", "rationale": "r"})


def test_decision_accepts_uuid_object_directly():
    item_id = uuid4()
    d = Decision.model_validate({"chosen": item_id, "rationale": "r"})
    assert d.chosen.id == item_id

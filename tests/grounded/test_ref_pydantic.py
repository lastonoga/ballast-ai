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


def test_decision_rewraps_ref_of_wrong_entity_type():
    """Cross-type Ref must NOT silently slip through — re-wrap to correct type."""
    other_id = uuid4()

    class Other(BaseModel):
        id: UUID
        name: str

    # Build a Ref[Other], try to assign into Ref[Item] field
    wrong_ref = Ref[Other](other_id)
    d = Decision.model_validate({"chosen": wrong_ref, "rationale": "r"})
    # Must be wrapped into Ref[Item], not stay as Ref[Other]
    assert d.chosen.entity_type is Item
    assert d.chosen.id == other_id


def test_decision_rejects_non_string_non_uuid_as_validation_error():
    """Non-UUID/non-string inputs must produce ValidationError, not raw TypeError."""
    with pytest.raises(ValidationError):
        Decision.model_validate({"chosen": 123, "rationale": "r"})
    with pytest.raises(ValidationError):
        Decision.model_validate({"chosen": [1, 2, 3], "rationale": "r"})
    with pytest.raises(ValidationError):
        Decision.model_validate({"chosen": None, "rationale": "r"})


def test_decision_json_schema_advertises_uuid_string():
    """JSON Schema must show 'string' type with 'uuid' format for the Ref field
    (needed by Tasks 11-13 for LLM-facing dynamic models)."""
    schema = Decision.model_json_schema()
    chosen_schema = schema["properties"]["chosen"]
    assert chosen_schema.get("type") == "string"
    assert chosen_schema.get("format") == "uuid"

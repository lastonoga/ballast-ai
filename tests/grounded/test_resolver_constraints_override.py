from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import GroundedResolver, Ref
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class Item(BaseModel):
    id: UUID
    name: str


class Ctx(BaseModel):
    items: list[Item]


class Out(BaseModel):
    chosen: Ref[Item]
    rationale: str


def test_constraints_override_restricts_to_subset():
    ids = [uuid4(), uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)])
    resolver = GroundedResolver(Out)

    # Override: only first ID allowed
    Dynamic, _ = resolver.build(ctx, constraints={"chosen": [ids[0]]})  # noqa: N806

    Dynamic.model_validate({"chosen": str(ids[0]), "rationale": "r"})
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": str(ids[1]), "rationale": "r"})


def test_constraints_override_with_unknown_path_errors():
    ids = [uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a")])
    resolver = GroundedResolver(Out)

    with pytest.raises(GroundedBuildError, match="unknown path"):
        resolver.build(ctx, constraints={"nonexistent_field": [ids[0]]})


def test_constraints_override_singleton_value():
    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)])
    resolver = GroundedResolver(Out)

    # Constraint is a scalar (single value rather than a list)
    Dynamic, _ = resolver.build(ctx, constraints={"chosen": ids[0]})  # noqa: N806
    Dynamic.model_validate({"chosen": str(ids[0]), "rationale": "r"})
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": str(ids[1]), "rationale": "r"})

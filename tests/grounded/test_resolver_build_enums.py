from typing import Literal, get_args
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Order(BaseModel):
    id: UUID
    status: Literal["draft", "ready", "sent", "cancelled"]


def test_literal_in_output_intersects_with_context_values():
    class Out(BaseModel):
        new_status: Literal["draft", "ready", "sent", "cancelled"]

    class Ctx(BaseModel):
        orders: list[Order]

    ctx = Ctx(orders=[
        Order(id=uuid4(), status="ready"),
        Order(id=uuid4(), status="sent"),
    ])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806
    # Only values actually present in context become allowed
    field_args = get_args(Dynamic.model_fields["new_status"].annotation)
    assert set(field_args) == {"ready", "sent"}


def test_literal_without_context_remains_unrestricted():
    class Out(BaseModel):
        new_status: Literal["draft", "ready", "sent", "cancelled"]

    class Ctx(BaseModel):
        unrelated: str

    ctx = Ctx(unrelated="x")
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806
    field_args = get_args(Dynamic.model_fields["new_status"].annotation)
    assert set(field_args) == {"draft", "ready", "sent", "cancelled"}


def test_literal_validation_rejects_unintersected_value():
    class Out(BaseModel):
        new_status: Literal["draft", "ready", "sent", "cancelled"]

    class Ctx(BaseModel):
        orders: list[Order]

    ctx = Ctx(orders=[Order(id=uuid4(), status="ready")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))  # noqa: N806

    with pytest.raises(ValidationError):
        Dynamic.model_validate({"new_status": "draft"})

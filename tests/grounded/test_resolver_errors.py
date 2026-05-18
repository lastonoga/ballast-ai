import warnings
from typing import Optional
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class Item(BaseModel):
    id: UUID
    name: str


def test_no_instances_raises_with_helpful_message():
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        unrelated: str

    ctx = Ctx(unrelated="x")
    with pytest.raises(GroundedBuildError) as exc_info:
        build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))
    assert "Item" in str(exc_info.value)
    assert "context" in str(exc_info.value).lower()


def test_optional_ref_with_no_instances_only_allows_none():
    class Out(BaseModel):
        maybe: Optional[Ref[Item]] = None  # noqa: UP007,UP045

    class Ctx(BaseModel):
        unrelated: str

    Dynamic = build_dynamic(  # noqa: N806
        Out, scan_output(Out), scan_context(Ctx(unrelated="x"), scan_output(Out))
    )
    # None must validate
    obj = Dynamic.model_validate({"maybe": None})
    assert obj.maybe is None


def test_large_allowed_set_emits_warning(recwarn):
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        items: list[Item]

    items = [Item(id=uuid4(), name=f"n{i}") for i in range(1500)]
    ctx = Ctx(items=items)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

    matched = [w for w in recwarn.list if "SemanticRouter" in str(w.message)]
    assert len(matched) >= 1

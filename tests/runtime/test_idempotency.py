from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ballast.runtime import IdempotencyInput


def test_accepts_allowed_primitive_types():
    inp = IdempotencyInput(
        namespace="test",
        parts={
            "str_field": "hello",
            "int_field": 42,
            "uuid_field": uuid4(),
            "dt_field": datetime.now(tz=UTC),
            "dec_field": Decimal("12.34"),
            "bool_field": True,
        },
    )
    assert inp.namespace == "test"


def test_rejects_float_values():
    with pytest.raises(ValidationError, match="float"):
        IdempotencyInput(namespace="test", parts={"bad": 1.5})


def test_rejects_unknown_object():
    class Custom:
        pass

    with pytest.raises(ValidationError):
        IdempotencyInput(namespace="test", parts={"bad": Custom()})


def test_is_frozen():
    inp = IdempotencyInput(namespace="t", parts={"a": 1})
    with pytest.raises(ValidationError):
        inp.namespace = "other"  # type: ignore[misc]


def test_canonical_json_is_stable_across_dict_orderings():
    a = IdempotencyInput(namespace="ns", parts={"x": 1, "y": 2})
    b = IdempotencyInput(namespace="ns", parts={"y": 2, "x": 1})
    assert a.canonical_json() == b.canonical_json()


def test_canonical_json_differs_for_different_inputs():
    a = IdempotencyInput(namespace="ns", parts={"x": 1})
    b = IdempotencyInput(namespace="ns", parts={"x": 2})
    assert a.canonical_json() != b.canonical_json()


def test_canonical_json_distinguishes_namespaces():
    a = IdempotencyInput(namespace="A", parts={"x": 1})
    b = IdempotencyInput(namespace="B", parts={"x": 1})
    assert a.canonical_json() != b.canonical_json()

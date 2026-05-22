from datetime import UTC, datetime
from uuid import UUID

import pytest

from ballast.runtime import Det, IdempotencyInput


@pytest.mark.asyncio
async def test_now_returns_timezone_aware_datetime():
    result = await Det.now()
    assert isinstance(result, datetime)
    assert result.tzinfo is UTC


@pytest.mark.asyncio
async def test_uuid4_returns_unique_uuids():
    a = await Det.uuid4()
    b = await Det.uuid4()
    assert isinstance(a, UUID)
    assert isinstance(b, UUID)
    assert a != b


@pytest.mark.asyncio
async def test_random_choice_returns_one_of_sequence():
    seq = ["a", "b", "c"]
    chosen = await Det.random_choice(seq)
    assert chosen in seq


@pytest.mark.asyncio
async def test_random_choice_empty_raises():
    with pytest.raises(IndexError):
        await Det.random_choice([])


@pytest.mark.asyncio
async def test_uuid_for_same_input_same_uuid():
    a = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"x": 1, "y": 2}))
    b = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"y": 2, "x": 1}))
    assert isinstance(a, UUID)
    assert a == b


@pytest.mark.asyncio
async def test_uuid_for_different_input_different_uuid():
    a = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"x": 1}))
    b = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"x": 2}))
    assert a != b


@pytest.mark.asyncio
async def test_uuid_for_different_namespace_different_uuid():
    a = await Det.uuid_for(IdempotencyInput(namespace="A", parts={"x": 1}))
    b = await Det.uuid_for(IdempotencyInput(namespace="B", parts={"x": 1}))
    assert a != b


@pytest.mark.asyncio
async def test_uuid_for_is_uuid5():
    a = await Det.uuid_for(IdempotencyInput(namespace="t", parts={"x": 1}))
    assert a.version == 5


@pytest.mark.asyncio
async def test_uuid_for_pinned_known_value():
    """Regression guard: any change to canonical_json serialization OR
    to the fixed _UUID_NAMESPACE will silently break determinism. This
    test pins one known input → known UUID. Update only with deliberate
    intent (e.g. namespace bump for backwards-incompat schema change)."""
    result = await Det.uuid_for(IdempotencyInput(namespace="t", parts={"x": 1}))
    assert result == UUID("17662462-f0fb-501e-a173-342eecebe6cd")


@pytest.mark.asyncio
async def test_random_choice_accepts_tuple():
    """random_choice must accept any Sequence, not just list."""
    chosen = await Det.random_choice(("a", "b", "c"))
    assert chosen in ("a", "b", "c")

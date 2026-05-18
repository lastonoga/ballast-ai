from datetime import UTC, datetime
from uuid import UUID

import pytest

from pydantic_ai_stateflow.runtime import Det, IdempotencyInput


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

from typing import ClassVar

import pytest

from pydantic_ai_stateflow.patterns import Pattern


class ConcretePattern:
    """Has all attributes Pattern protocol requires."""

    name: ClassVar[str] = "concrete"

    async def run(self, input: int) -> int:
        return input * 2


class WrongPattern:
    """Missing `run` method."""

    name: ClassVar[str] = "wrong"


def test_concrete_pattern_satisfies_protocol():
    instance: Pattern[int, int] = ConcretePattern()
    assert instance.name == "concrete"


def test_wrong_pattern_does_not_satisfy_protocol_at_runtime_when_checked():
    # Pattern is `@runtime_checkable`; isinstance check enforces structure.
    assert isinstance(ConcretePattern(), Pattern)
    assert not isinstance(WrongPattern(), Pattern)


@pytest.mark.asyncio
async def test_pattern_run_returns_expected():
    p = ConcretePattern()
    result = await p.run(5)
    assert result == 10

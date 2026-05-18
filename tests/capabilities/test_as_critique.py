import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.capabilities.helpers import Critique, as_critique


class CustomVerdict(BaseModel):
    passed: bool
    issues: list[str] = []


@pytest.mark.asyncio
async def test_as_critique_wraps_async_function():
    async def fn(payload):
        return Critique(passed=True, confidence=1.0)

    agent = as_critique(fn)
    result = await agent.run("anything")
    assert result.output.passed is True


@pytest.mark.asyncio
async def test_as_critique_wraps_object_with_check_method():
    class C:
        async def check(self, payload):
            return Critique(passed=False, issues=["bad"])

    agent = as_critique(C())
    result = await agent.run("anything")
    assert result.output.passed is False
    assert result.output.issues == ["bad"]


@pytest.mark.asyncio
async def test_as_critique_coerces_custom_pass_object():
    """A return object with .passed coerces into Critique."""

    async def fn(payload):
        return CustomVerdict(passed=True, issues=["minor"])

    agent = as_critique(fn)
    result = await agent.run("anything")
    assert result.output.passed is True
    assert result.output.issues == ["minor"]


@pytest.mark.asyncio
async def test_as_critique_coerces_bool():
    async def fn(payload):
        return True

    agent = as_critique(fn)
    result = await agent.run("anything")
    assert result.output.passed is True

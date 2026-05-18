import pytest

from pydantic_ai_stateflow.providers import CoreProvider
from pydantic_ai_stateflow.runtime import Det
from pydantic_ai_stateflow.runtime.container import DefaultContainer


@pytest.mark.asyncio
async def test_core_provider_binds_det() -> None:
    container = DefaultContainer()
    provider = CoreProvider()
    await provider.register(container)

    result = container.get(type(Det))
    assert result is Det

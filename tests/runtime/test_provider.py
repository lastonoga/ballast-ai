import pytest

from pydantic_ai_stateflow.runtime.container import Container, DefaultContainer
from pydantic_ai_stateflow.runtime.provider import ServiceProvider


class Greeter:
    def hello(self) -> str:
        return "hello"


class GreeterProvider:
    async def register(self, container: Container) -> None:
        container.bind(Greeter, lambda _: Greeter())


@pytest.mark.asyncio
async def test_provider_protocol_register_binds_into_container():
    c = DefaultContainer()
    p: ServiceProvider = GreeterProvider()
    await p.register(c)
    assert isinstance(c.get(Greeter), Greeter)


@pytest.mark.asyncio
async def test_concrete_provider_satisfies_protocol():
    assert isinstance(GreeterProvider(), ServiceProvider)

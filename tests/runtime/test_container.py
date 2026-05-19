from typing import Protocol

import pytest

from pydantic_ai_stateflow.runtime.container import Container, DefaultContainer


class Greeter(Protocol):
    def hello(self) -> str: ...


class ConcreteGreeter:
    def hello(self) -> str:
        return "hi"


def test_bind_and_get_singleton():
    c = DefaultContainer()
    c.bind(Greeter, lambda _: ConcreteGreeter())
    g = c.get(Greeter)
    assert g.hello() == "hi"


def test_singleton_returns_same_instance():
    c = DefaultContainer()
    c.bind(Greeter, lambda _: ConcreteGreeter())
    assert c.get(Greeter) is c.get(Greeter)


def test_non_singleton_returns_fresh_instance():
    c = DefaultContainer()
    c.bind(Greeter, lambda _: ConcreteGreeter(), singleton=False)
    assert c.get(Greeter) is not c.get(Greeter)


def test_get_unknown_type_raises_key_error():
    c = DefaultContainer()
    with pytest.raises(KeyError, match="Greeter"):
        c.get(Greeter)


def test_factory_receives_container_for_dependencies():
    """Factory can resolve other deps via the container parameter."""

    class Foo:
        pass

    class Bar:
        def __init__(self, foo: Foo):
            self.foo = foo

    c = DefaultContainer()
    c.bind(Foo, lambda _: Foo())
    c.bind(Bar, lambda c: Bar(c.get(Foo)))

    assert isinstance(c.get(Bar).foo, Foo)


def test_default_container_satisfies_protocol():
    assert isinstance(DefaultContainer(), Container)


# --- F15: bind/get extensions -------------------------------------------


def test_bind_and_get_returns_instance():
    """Pre-built instance binding round-trips via get."""
    c = DefaultContainer()
    instance = ConcreteGreeter()
    c.bind(Greeter, instance)
    assert c.get(Greeter) is instance


def test_bind_callable_factory_returns_fresh_per_call_when_singleton_false():
    c = DefaultContainer()
    c.bind(Greeter, lambda: ConcreteGreeter(), singleton=False)
    a = c.get(Greeter)
    b = c.get(Greeter)
    assert a is not b
    assert a.hello() == "hi"


def test_bind_callable_singleton_caches():
    c = DefaultContainer()
    c.bind(Greeter, lambda: ConcreteGreeter())  # singleton=True default
    assert c.get(Greeter) is c.get(Greeter)


async def test_aget_awaits_async_factory():
    c = DefaultContainer()

    async def _make() -> ConcreteGreeter:
        return ConcreteGreeter()

    c.bind(Greeter, _make)
    g = await c.aget(Greeter)
    assert g.hello() == "hi"
    # Singleton caching survives across aget calls.
    g2 = await c.aget(Greeter)
    assert g is g2


def test_get_raises_runtime_error_on_async_binding():
    c = DefaultContainer()

    async def _make() -> ConcreteGreeter:
        return ConcreteGreeter()

    c.bind(Greeter, _make)
    with pytest.raises(RuntimeError, match="async"):
        c.get(Greeter)


def test_get_raises_keyerror_when_unbound():
    c = DefaultContainer()
    with pytest.raises(KeyError, match="Greeter"):
        c.get(Greeter)


def test_has_reflects_binding():
    c = DefaultContainer()
    assert not c.has(Greeter)
    c.bind(Greeter, ConcreteGreeter())
    assert c.has(Greeter)


def test_rebind_replaces_previous_value():
    c = DefaultContainer()
    first = ConcreteGreeter()
    second = ConcreteGreeter()
    c.bind(Greeter, first)
    c.bind(Greeter, second)
    assert c.get(Greeter) is second


async def test_aget_non_singleton_async_factory_returns_fresh():
    c = DefaultContainer()

    async def _make() -> ConcreteGreeter:
        return ConcreteGreeter()

    c.bind(Greeter, _make, singleton=False)
    a = await c.aget(Greeter)
    b = await c.aget(Greeter)
    assert a is not b


async def test_aget_works_on_sync_factory_and_instance():
    """aget is the safe universal accessor — works for sync bindings too."""
    c = DefaultContainer()
    inst = ConcreteGreeter()
    c.bind(Greeter, inst)
    assert await c.aget(Greeter) is inst

    c2 = DefaultContainer()
    c2.bind(Greeter, lambda: ConcreteGreeter())
    g = await c2.aget(Greeter)
    assert g.hello() == "hi"

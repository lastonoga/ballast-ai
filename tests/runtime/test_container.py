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

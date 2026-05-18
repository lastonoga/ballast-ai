from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Container(Protocol):
    """Type-keyed DI registry.

    Bindings are keyed by Protocol or concrete type. Factories receive the
    Container so they can resolve their own dependencies.

    No string keys (per spec 4A.0.7 — service-locator pattern forbidden).
    """

    def bind(
        self,
        protocol: type[T],
        factory: Callable[[Container], T],
        *,
        singleton: bool = True,
    ) -> None: ...

    def get(self, protocol: type[T]) -> T: ...


class DefaultContainer:
    """Minimal type-keyed DI container.

    Singleton by default — `singleton=False` for fresh-per-resolution
    bindings (rare; only needed for stateful per-request services).
    """

    def __init__(self) -> None:
        self._factories: dict[type, tuple[Callable[[Container], Any], bool]] = {}
        self._instances: dict[type, Any] = {}

    def bind(
        self,
        protocol: type[T],
        factory: Callable[[Container], T],
        *,
        singleton: bool = True,
    ) -> None:
        self._factories[protocol] = (factory, singleton)
        # Drop any previously-cached singleton if rebinding
        self._instances.pop(protocol, None)

    def get(self, protocol: type[T]) -> T:
        if protocol not in self._factories:
            raise KeyError(f"No binding for {protocol.__name__}")
        factory, singleton = self._factories[protocol]
        if singleton:
            if protocol not in self._instances:
                self._instances[protocol] = factory(self)
            return cast(T, self._instances[protocol])
        return cast(T, factory(self))

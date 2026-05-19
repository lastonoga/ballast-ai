from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias, TypeVar, Union, cast, runtime_checkable

T = TypeVar("T")

# Public binding-value union:
#   - a pre-built instance of T
#   - a zero-arg factory returning T (sync or async)
#   - the legacy `Callable[[Container], T]` form (kept for back-compat
#     with CoreProvider / PersistenceProvider and existing app code)
#
# Declared as a generic Union (rather than a 3.12 ``type BindValue[T] = ...``
# alias) to stay on the 3.11 baseline. Indexing by a concrete type works
# at call sites but is not type-checked structurally — callers should pass
# the value directly to ``bind`` and let inference flow from the protocol
# argument.
BindValue: TypeAlias = Union[  # noqa: UP007 — keep Union for 3.11 mypy alias support
    T,
    Callable[[], T],
    Callable[["Container"], T],
    Callable[[], Awaitable[T]],
]


@runtime_checkable
class Container(Protocol):
    """Type-keyed DI registry.

    Bindings are keyed by Protocol or concrete type. Values can be:

    * a pre-built instance (``container.bind(Repo, InMemoryRepo())``),
    * a zero-arg factory (sync or async),
    * the legacy container-arg factory ``lambda c: ...`` used by built-in
      providers (kept for back-compat — apps should prefer the simpler
      forms above).

    No string keys (per spec 4A.0.7 — service-locator pattern forbidden).
    """

    def bind(
        self,
        protocol: type[T],
        value: BindValue[T],
        *,
        singleton: bool = True,
    ) -> None: ...

    def get(self, protocol: type[T]) -> T: ...

    async def aget(self, protocol: type[T]) -> T: ...

    def has(self, protocol: type) -> bool: ...


class DefaultContainer:
    """Minimal type-keyed DI container.

    Singleton by default — pass ``singleton=False`` for fresh-per-resolve
    bindings (rare; only for stateful per-request services).

    Async factories must be resolved via :meth:`aget`. Calling :meth:`get`
    on an async binding raises :class:`RuntimeError` with a clear message.
    """

    # Resolver shape: (callable_taking_container_or_none, is_async, singleton)
    # If resolver is None the value is a pre-built instance stored in `_instances`.
    _Resolver = tuple[Callable[["Container"], Any] | None, bool, bool]

    def __init__(self) -> None:
        self._resolvers: dict[type, DefaultContainer._Resolver] = {}
        self._instances: dict[type, Any] = {}

    # ----- bind -----------------------------------------------------------

    def bind(
        self,
        protocol: type[T],
        value: BindValue[T],
        *,
        singleton: bool = True,
    ) -> None:
        """Register a binding for ``protocol``.

        Accepts a pre-built instance, a zero-arg factory (sync or async),
        or the legacy ``Callable[[Container], T]`` factory.
        """
        # Drop any previously-cached singleton if rebinding.
        self._instances.pop(protocol, None)
        self._resolvers.pop(protocol, None)

        if not callable(value):
            # Pre-built instance — cache verbatim, no resolver needed.
            self._instances[protocol] = value
            self._resolvers[protocol] = (None, False, True)
            return

        is_async = inspect.iscoroutinefunction(value)
        # Inspect arity so we know whether to pass the container in.
        try:
            sig = inspect.signature(value)
            # Count positional-ish params without defaults.
            required = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and p.default is inspect.Parameter.empty
            ]
            takes_container = len(required) >= 1
        except (TypeError, ValueError):
            # Builtins / C-callables / classes with weird __init__ — assume
            # zero-arg. The user can always pass a lambda to disambiguate.
            takes_container = False

        resolver: Callable[[Container], Any]
        if takes_container:
            resolver = cast(Callable[[Container], Any], value)
        else:
            zero_arg = cast(Callable[[], Any], value)

            def _resolver(_c: Container, _f: Callable[[], Any] = zero_arg) -> Any:
                return _f()

            resolver = _resolver

        self._resolvers[protocol] = (resolver, is_async, singleton)

    # ----- get / aget -----------------------------------------------------

    def get(self, protocol: type[T]) -> T:
        if protocol not in self._resolvers:
            raise KeyError(f"No binding for {protocol.__name__}")
        resolver, is_async, singleton = self._resolvers[protocol]
        if is_async:
            raise RuntimeError(
                f"binding for {protocol.__name__} is async; "
                f"use `await container.aget({protocol.__name__})` instead"
            )
        if resolver is None:
            # Pre-built instance binding.
            return cast(T, self._instances[protocol])
        if singleton:
            if protocol not in self._instances:
                self._instances[protocol] = resolver(self)
            return cast(T, self._instances[protocol])
        return cast(T, resolver(self))

    async def aget(self, protocol: type[T]) -> T:
        if protocol not in self._resolvers:
            raise KeyError(f"No binding for {protocol.__name__}")
        resolver, is_async, singleton = self._resolvers[protocol]
        if resolver is None:
            return cast(T, self._instances[protocol])
        if singleton and protocol in self._instances:
            return cast(T, self._instances[protocol])
        result = resolver(self)
        if is_async:
            result = await result
        if singleton:
            self._instances[protocol] = result
        return cast(T, result)

    # ----- has ------------------------------------------------------------

    def has(self, protocol: type) -> bool:
        return protocol in self._resolvers

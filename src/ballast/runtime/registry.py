"""Generic named-item registry.

Type-safe collection keyed by an item's ``name: str`` attribute. Apps
use this for agent dispatch, workflow dispatch, plugin registries —
any "map a string identifier to an object" need.

SOLID rationale:
- **Single Responsibility**: store + retrieve by name. Nothing else.
- **Open/Closed**: extend by registering new items; never modify the
  registry's behaviour.
- **Liskov**: any ``T`` satisfying ``Named`` (i.e. has ``name: str``)
  works — concrete agent class, mock, subclass, all interchangeable.
- **Interface Segregation**: the ``Named`` Protocol exposes ONE
  attribute — apps don't pay for unused interface surface.
- **Dependency Inversion**: callers depend on ``Registry[Named]``
  (abstraction), not a concrete dict or app-specific dispatch helper.

Example::

    from ballast import Registry

    agents: Registry[BallastAgent] = Registry()
    agents.register(notes_agent)
    agents.register(approval_agent)

    # In an HTTP handler:
    agent = agents.get(thread.agent)
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class Named(Protocol):
    """Anything with a ``name: str`` attribute."""

    name: str


T = TypeVar("T", bound=Named)


class Registry(Generic[T]):
    """Type-safe named-item collection.

    Items must expose ``name: str``. ``register`` enforces uniqueness;
    ``override`` allows replacement (useful for tests). ``get`` is
    strict — missing names raise ``KeyError``.
    """

    def __init__(self, *items: T) -> None:
        """Construct with optional initial items.

        ``Registry(a, b, c)`` is shorthand for an empty registry plus
        three ``register`` calls.
        """
        self._items: dict[str, T] = {}
        for item in items:
            self.register(item)

    def register(self, item: T) -> T:
        """Add ``item`` to the registry. Returns the item for chaining.

        Raises ``ValueError`` if an item with the same ``name`` is
        already registered. Use ``override`` for replace-on-duplicate
        semantics (e.g. tests swapping in a mock).
        """
        name = self._extract_name(item)
        if name in self._items:
            raise ValueError(
                f"Duplicate registration for name {name!r}: "
                f"existing={self._items[name]!r}, new={item!r}. "
                f"Use ``override`` to replace.",
            )
        self._items[name] = item
        return item

    def override(self, item: T) -> T | None:
        """Replace any existing item under the same ``name``.

        Returns the previous item (or ``None`` if no prior registration).
        Useful for test fixtures that want to swap in a mock for a
        single test then restore the original.
        """
        name = self._extract_name(item)
        previous = self._items.get(name)
        self._items[name] = item
        return previous

    def get(self, name: str) -> T:
        """Look up by ``name``. Raises ``KeyError`` if missing."""
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(
                f"No item registered under name {name!r}. "
                f"Known: {sorted(self._items.keys())!r}",
            ) from exc

    def remove(self, name: str) -> T:
        """Drop the registration and return the item.

        Raises ``KeyError`` if missing.
        """
        try:
            return self._items.pop(name)
        except KeyError as exc:
            raise KeyError(
                f"No item registered under name {name!r}",
            ) from exc

    def names(self) -> list[str]:
        """Sorted list of registered names."""
        return sorted(self._items.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.names()!r})"

    @staticmethod
    def _extract_name(item: T) -> str:
        name = getattr(item, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"Item {item!r} must expose a non-empty ``name: str`` "
                f"attribute to register; got name={name!r}",
            )
        return name


__all__ = ["Named", "Registry"]

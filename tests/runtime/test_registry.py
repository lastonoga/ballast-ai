"""Unit tests for the generic ``Registry``."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pydantic_ai_stateflow.runtime.registry import Registry


@dataclass
class _Named:
    name: str
    value: int = 0


def test_register_adds_item() -> None:
    r: Registry[_Named] = Registry()
    a = _Named(name="a", value=1)
    r.register(a)
    assert r.get("a") is a
    assert "a" in r


def test_register_returns_item_for_chaining() -> None:
    r: Registry[_Named] = Registry()
    a = _Named(name="a")
    assert r.register(a) is a


def test_constructor_accepts_initial_items() -> None:
    a = _Named(name="a")
    b = _Named(name="b")
    r = Registry(a, b)
    assert r.get("a") is a
    assert r.get("b") is b
    assert r.names() == ["a", "b"]


def test_register_duplicate_raises() -> None:
    r: Registry[_Named] = Registry()
    r.register(_Named(name="a"))
    with pytest.raises(ValueError, match="Duplicate registration"):
        r.register(_Named(name="a"))


def test_override_replaces_existing() -> None:
    r: Registry[_Named] = Registry()
    a1 = _Named(name="a", value=1)
    a2 = _Named(name="a", value=2)
    r.register(a1)
    previous = r.override(a2)
    assert previous is a1
    assert r.get("a") is a2


def test_override_first_time_returns_none() -> None:
    r: Registry[_Named] = Registry()
    a = _Named(name="a")
    assert r.override(a) is None
    assert r.get("a") is a


def test_get_missing_raises_keyerror() -> None:
    r: Registry[_Named] = Registry()
    with pytest.raises(KeyError, match="No item registered"):
        r.get("missing")


def test_remove_returns_item() -> None:
    r: Registry[_Named] = Registry()
    a = _Named(name="a")
    r.register(a)
    assert r.remove("a") is a
    assert "a" not in r


def test_remove_missing_raises() -> None:
    r: Registry[_Named] = Registry()
    with pytest.raises(KeyError):
        r.remove("missing")


def test_iter_yields_items() -> None:
    a = _Named(name="a")
    b = _Named(name="b")
    r = Registry(a, b)
    items = list(r)
    assert a in items
    assert b in items
    assert len(items) == 2


def test_len() -> None:
    r: Registry[_Named] = Registry()
    assert len(r) == 0
    r.register(_Named(name="a"))
    assert len(r) == 1


def test_register_rejects_item_without_name() -> None:
    r: Registry[_Named] = Registry()

    class NoName:
        pass

    with pytest.raises(TypeError, match="non-empty ``name: str``"):
        r.register(NoName())  # type: ignore[arg-type]


def test_register_rejects_item_with_empty_name() -> None:
    r: Registry[_Named] = Registry()
    with pytest.raises(TypeError, match="non-empty ``name: str``"):
        r.register(_Named(name=""))


def test_names_sorted() -> None:
    r = Registry(_Named(name="b"), _Named(name="a"), _Named(name="c"))
    assert r.names() == ["a", "b", "c"]

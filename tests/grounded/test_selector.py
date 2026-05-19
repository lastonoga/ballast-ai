"""Tests for the Selector primitive + SelectorRegistry + extract_selector."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.grounded.ref import Ref
from pydantic_ai_stateflow.grounded.selector import (
    Selector,
    SelectorRegistry,
    extract_selector,
    resolve_selector_ids,
)


class _Note(BaseModel):
    id: UUID
    title: str


def test_selector_inline_is_callable() -> None:
    sel = Selector(lambda _ctx: [uuid4()])
    assert not sel.is_named()
    fn = sel.resolve(registry=None)
    assert callable(fn)


def test_selector_named_resolves_via_registry() -> None:
    ids = [uuid4()]
    reg = SelectorRegistry()
    reg.register("open", lambda _ctx: ids)

    sel = Selector("open")
    assert sel.is_named()
    fn = sel.resolve(reg)
    assert fn(None) == ids


def test_selector_named_requires_registry() -> None:
    sel = Selector("missing")
    with pytest.raises(KeyError, match="missing"):
        sel.resolve(registry=None)


def test_registry_rejects_duplicate_names() -> None:
    reg = SelectorRegistry()
    reg.register("x", lambda _c: [])
    with pytest.raises(ValueError, match="already registered"):
        reg.register("x", lambda _c: [])


def test_registry_get_unknown_lists_available() -> None:
    reg = SelectorRegistry()
    reg.register("a", lambda _c: [])
    reg.register("b", lambda _c: [])
    with pytest.raises(KeyError) as exc:
        reg.get("zzz")
    assert "['a', 'b']" in str(exc.value)


def test_extract_selector_from_annotated_ref() -> None:
    sel = Selector(lambda _c: [])
    ann = Annotated[Ref[_Note], sel]
    target, found = extract_selector(ann)
    assert target is _Note
    assert found is sel


def test_extract_selector_from_bare_ref() -> None:
    target, found = extract_selector(Ref[_Note])
    assert target is _Note
    assert found is None


def test_extract_selector_unrecognized_type() -> None:
    target, found = extract_selector(int)
    assert target is None
    assert found is None


async def test_resolve_selector_ids_from_uuids() -> None:
    ids = [uuid4(), uuid4()]
    sel = Selector(lambda _c: ids)
    out = await resolve_selector_ids(sel, ctx=None, registry=None)
    assert out == ids


async def test_resolve_selector_ids_from_entities() -> None:
    n1 = _Note(id=uuid4(), title="a")
    n2 = _Note(id=uuid4(), title="b")
    sel = Selector(lambda _c: [n1, n2])
    out = await resolve_selector_ids(sel, ctx=None, registry=None)
    assert out == [n1.id, n2.id]


async def test_resolve_selector_ids_supports_async_selector() -> None:
    ids = [uuid4()]

    async def _async_sel(_ctx: object) -> list[UUID]:
        return ids

    out = await resolve_selector_ids(Selector(_async_sel), ctx=None, registry=None)
    assert out == ids


async def test_resolve_selector_ids_rejects_non_id_objects() -> None:
    sel = Selector(lambda _c: ["not-an-id"])
    with pytest.raises(TypeError, match="without an .id attribute"):
        await resolve_selector_ids(sel, ctx=None, registry=None)

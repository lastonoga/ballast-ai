"""``CoALABase`` ABC — default observe/learn; abstract retrieve/act."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase, CoALAUnit


class _Minimal(CoALABase[str, str, dict, str]):
    async def retrieve(self, observation): return {"data": "ctx"}
    async def act(self, observation, context): return f"acted on {observation}"


def test_default_observe_is_identity() -> None:
    import asyncio
    out = asyncio.run(_Minimal().observe("hello"))
    assert out == "hello"


def test_default_learn_is_no_op() -> None:
    import asyncio
    result = asyncio.run(_Minimal().learn("o", {}, "out"))
    assert result is None


def test_subclass_satisfies_coala_unit_protocol() -> None:
    assert isinstance(_Minimal(), CoALAUnit)


def test_abstract_retrieve_must_be_overridden() -> None:
    class _NoRetrieve(CoALABase):
        async def act(self, observation, context): return None

    with pytest.raises(TypeError, match="abstract"):
        _NoRetrieve()  # type: ignore[abstract]


def test_abstract_act_must_be_overridden() -> None:
    class _NoAct(CoALABase):
        async def retrieve(self, observation): return {}

    with pytest.raises(TypeError, match="abstract"):
        _NoAct()  # type: ignore[abstract]

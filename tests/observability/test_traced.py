from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from pydantic_ai_stateflow.observability.spans import traced


@pytest.mark.asyncio
async def test_traced_passthrough_when_logfire_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "logfire", None)

    @traced("test.span")
    async def fn(x: int) -> int:
        return x * 2

    assert await fn(3) == 6


@pytest.mark.asyncio
async def test_traced_emits_span_when_logfire_present(monkeypatch):
    fake = MagicMock()
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=ctx_mgr)
    ctx_mgr.__exit__ = MagicMock(return_value=False)
    fake.span = MagicMock(return_value=ctx_mgr)
    monkeypatch.setitem(sys.modules, "logfire", fake)

    @traced("test.span")
    async def fn() -> str:
        return "ok"

    assert await fn() == "ok"
    fake.span.assert_called_once()
    args, _ = fake.span.call_args
    assert args[0] == "test.span"


@pytest.mark.asyncio
async def test_traced_attaches_attributes_lambda(monkeypatch):
    fake = MagicMock()
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=ctx_mgr)
    ctx_mgr.__exit__ = MagicMock(return_value=False)
    fake.span = MagicMock(return_value=ctx_mgr)
    monkeypatch.setitem(sys.modules, "logfire", fake)

    @traced("test.span", attrs=lambda x: {"x": x})
    async def fn(x: int) -> int:
        return x

    await fn(42)
    _, kwargs = fake.span.call_args
    assert kwargs == {"x": 42}


@pytest.mark.asyncio
async def test_traced_propagates_exceptions(monkeypatch):
    monkeypatch.setitem(sys.modules, "logfire", None)

    @traced("test.span")
    async def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await boom()

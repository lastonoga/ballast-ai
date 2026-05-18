from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from pydantic_ai_stateflow.observability import ObservabilityProvider, has_logfire
from pydantic_ai_stateflow.runtime import Engine


class _Spy:
    def __init__(self) -> None:
        self.calls = 0

    async def register(self, _container: object) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_provider_is_noop_when_logfire_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "logfire", None)  # simulate absence
    engine = Engine(providers=[ObservabilityProvider(service_name="t")])
    await engine.boot()
    assert engine._booted is True  # boots successfully despite no logfire


@pytest.mark.asyncio
async def test_provider_calls_logfire_configure_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.configure = MagicMock()
    fake.instrument_pydantic_ai = MagicMock()
    fake.instrument_httpx = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    engine = Engine(
        providers=[
            ObservabilityProvider(service_name="svc", environment="test"),
        ],
    )
    await engine.boot()
    fake.configure.assert_called_once()
    kwargs = fake.configure.call_args.kwargs
    assert kwargs["service_name"] == "svc"
    assert kwargs["environment"] == "test"


@pytest.mark.asyncio
async def test_provider_instruments_pydantic_ai_and_httpx_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    engine = Engine(
        providers=[
            ObservabilityProvider(
                service_name="svc",
                instrument_pydantic_ai=True,
                instrument_httpx=True,
            ),
        ],
    )
    await engine.boot()
    fake.instrument_pydantic_ai.assert_called_once()
    fake.instrument_httpx.assert_called_once()


@pytest.mark.asyncio
async def test_provider_first_invariant_fails_if_other_provider_already_registered() -> None:
    """Spec 4H — ObservabilityProvider must be first in the list."""
    from pydantic_ai_stateflow import EngineInvariantViolation

    engine = Engine(providers=[_Spy(), ObservabilityProvider(service_name="t")])
    with pytest.raises(EngineInvariantViolation, match="first"):
        await engine.boot()


def test_has_logfire_returns_bool() -> None:
    assert isinstance(has_logfire(), bool)


@pytest.mark.asyncio
async def test_provider_skips_instrument_fastapi_unless_app_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    engine = Engine(
        providers=[
            ObservabilityProvider(service_name="svc"),
        ],
    )
    await engine.boot()
    fake.instrument_fastapi.assert_not_called()

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

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


# ---------------------------------------------------------------------------
# Diagnostic logging — surfaces logfire status to the operator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_logs_when_logfire_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(sys.modules, "logfire", None)
    engine = Engine(providers=[ObservabilityProvider(service_name="t")])
    with caplog.at_level(logging.INFO, logger="pydantic_ai_stateflow"):
        await engine.boot()
    assert any(
        "logfire not installed" in rec.getMessage() for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


@pytest.mark.asyncio
async def test_provider_logs_token_present_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    monkeypatch.setenv("LOGFIRE_TOKEN", "secret-token-abc")
    engine = Engine(
        providers=[ObservabilityProvider(service_name="svc", environment="prod")],
    )
    with caplog.at_level(logging.INFO, logger="pydantic_ai_stateflow"):
        await engine.boot()
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("token=present" in m for m in msgs), msgs
    assert not any("token=MISSING" in m for m in msgs), msgs


@pytest.mark.asyncio
async def test_provider_logs_token_missing_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    engine = Engine(
        providers=[ObservabilityProvider(service_name="svc", environment="dev")],
    )
    with caplog.at_level(logging.INFO, logger="pydantic_ai_stateflow"):
        await engine.boot()
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("token=MISSING" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# instrument_app — post-construction FastAPI wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_instrument_app_calls_logfire_instrument_fastapi_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    provider = ObservabilityProvider(service_name="svc")
    # Provider must be registered first so the logfire soft-import path
    # has run (configure called etc.). The new ``instrument_app`` method
    # is independent though — it works even before register.
    engine = Engine(providers=[provider])
    await engine.boot()
    app = FastAPI()
    provider.instrument_app(app)
    provider.instrument_app(app)  # idempotent
    fake.instrument_fastapi.assert_called_once_with(app)


@pytest.mark.asyncio
async def test_instrument_app_noop_when_logfire_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "logfire", None)
    provider = ObservabilityProvider(service_name="svc")
    # Must not raise.
    provider.instrument_app(FastAPI())


@pytest.mark.asyncio
async def test_instrument_app_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    provider = ObservabilityProvider(service_name="svc", instrument_fastapi=False)
    engine = Engine(providers=[provider])
    await engine.boot()
    provider.instrument_app(FastAPI())
    fake.instrument_fastapi.assert_not_called()


def test_engine_fastapi_app_calls_instrument_app_on_observability_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine.fastapi_app walks providers and instruments the app."""
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    provider = ObservabilityProvider(service_name="svc")
    engine = Engine(providers=[provider])
    app = engine.fastapi_app()
    fake.instrument_fastapi.assert_called_once_with(app)


def test_engine_fastapi_app_skips_instrument_app_when_no_observability_provider() -> None:
    """No ObservabilityProvider → engine.fastapi_app stays silent."""

    class _Noop:
        async def register(self, _container: object) -> None:
            return None

    engine = Engine(providers=[_Noop()])
    # Should not raise (and obviously must not need logfire).
    engine.fastapi_app()

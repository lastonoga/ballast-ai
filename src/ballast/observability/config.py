"""``ObservabilityConfig`` — replacement for the deleted
``ObservabilityProvider`` class.

A plain dataclass holding the knobs that used to be Provider ctor args.
``install()`` configures Logfire + instrumentation once per process
(global Logfire SDK side effects). ``instrument_app(app)`` attaches
the FastAPI integration to a specific FastAPI app instance.

Idempotent: calling ``install()`` twice with the same config is a no-op;
calling with a different config raises ``RuntimeError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ballast.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

_logger = get_logger(__name__)

# Process-wide singleton tracking — like ObservabilityProvider used to
# track via the Container; here we use a module-level flag so the
# "called once" invariant is enforced without Container ceremony.
_installed_config: "ObservabilityConfig | None" = None


@dataclass(frozen=True)
class ObservabilityConfig:
    """Defaults for framework observability.

    Honors ``LOGFIRE_TOKEN`` env var (the logfire SDK reads it directly
    when ``logfire.configure()`` is called without an explicit token).
    No token → logfire emits warnings + becomes a no-op pipeline.
    """

    service_name: str = "pydantic-ai-stateflow"
    environment: str = "dev"
    instrument_pydantic_ai: bool = True
    instrument_httpx: bool = True
    instrument_fastapi: bool = True

    def install(self) -> None:
        """Configure Logfire + global instrumentation. Idempotent per process.

        Safe to call before any FastAPI app exists. Apps that want
        ``instrument_fastapi`` enabled must additionally call
        ``instrument_app(app)`` once the app is constructed.
        """
        global _installed_config
        if _installed_config is not None:
            if _installed_config == self:
                _logger.debug("ObservabilityConfig.install() already applied with same config; no-op")
                return
            raise RuntimeError(
                "ObservabilityConfig.install() already called with a different "
                f"config (old={_installed_config!r}, new={self!r}). Per-process "
                "Logfire setup is global; pick one config and stick with it.",
            )

        try:
            import logfire
        except ImportError:
            _logger.warning(
                "logfire not installed — observability is a no-op. "
                "`uv add logfire` to enable.",
            )
            _installed_config = self
            return

        # ``logfire.configure`` reads ``LOGFIRE_TOKEN`` from env when no
        # explicit token kwarg is supplied. We don't pass one — apps
        # that need a different secret-resolution path can pre-configure
        # logfire themselves and skip this call.
        try:
            logfire.configure(
                service_name=self.service_name,
                environment=self.environment,
                send_to_logfire="if-token-present",
            )
        except Exception:
            _logger.exception("logfire.configure failed; instrumentation skipped")
            _installed_config = self
            return

        if self.instrument_pydantic_ai:
            try:
                logfire.instrument_pydantic_ai()
            except Exception:
                _logger.exception("logfire.instrument_pydantic_ai failed")

        if self.instrument_httpx:
            try:
                logfire.instrument_httpx()
            except Exception:
                _logger.exception(
                    "logfire.instrument_httpx failed — install "
                    "logfire[httpx] to enable",
                )

        _installed_config = self
        _logger.info(
            "observability.install: configured (service=%s env=%s)",
            self.service_name, self.environment,
        )

    def instrument_app(self, app: "FastAPI") -> None:
        """Attach FastAPI-specific instrumentation to ``app``.

        Called AFTER ``install()`` (which configures the Logfire pipeline)
        and AFTER the FastAPI app is constructed. ``ballast.create_app()`` calls
        this automatically when ``observability.instrument_fastapi`` is
        True.
        """
        if not self.instrument_fastapi:
            return
        try:
            import logfire
        except ImportError:
            return
        try:
            logfire.instrument_fastapi(app)
        except Exception:
            _logger.exception(
                "logfire.instrument_fastapi failed — install "
                "logfire[fastapi] to enable",
            )


def _reset_observability_for_tests() -> None:
    """Test-only: drop the installed-config singleton."""
    global _installed_config
    _installed_config = None


def is_configured() -> bool:
    """True iff :meth:`ObservabilityConfig.install` has been called.

    Callers that emit logfire spans should gate on this to avoid
    ``LogfireNotConfiguredWarning`` in environments where logfire is
    importable but ``configure()`` was never invoked (e.g. tests).
    """
    return _installed_config is not None


__all__ = ["ObservabilityConfig", "is_configured"]

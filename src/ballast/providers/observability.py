"""ObservabilityProvider — installs Logfire + instrumentation from settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ballast.app import Ballast
    from ballast.observability.config import ObservabilityConfig


class ObservabilityProvider:
    """Install :class:`ObservabilityConfig` during ``.use(...)``.

    Pulls knobs from ``ballast.settings.observability`` by default; pass
    an explicit :class:`ObservabilityConfig` to override. Idempotent
    (the underlying ``install()`` is a no-op when called twice with the
    same config).
    """

    def __init__(self, config: "ObservabilityConfig | None" = None) -> None:
        self._config = config

    def register(self, ballast: "Ballast") -> None:
        from ballast.observability.config import ObservabilityConfig

        config = self._config
        if config is None:
            s = ballast.settings.observability
            config = ObservabilityConfig(
                service_name=s.service_name,
                environment=s.environment,
                instrument_pydantic_ai=s.instrument_pydantic_ai,
                instrument_httpx=s.instrument_httpx,
                instrument_fastapi=s.instrument_fastapi,
            )
        config.install()
        ballast._mark_observability_installed()


__all__ = ["ObservabilityProvider"]

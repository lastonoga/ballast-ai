"""DBOSProvider — wires the DBOSConfig (from settings or explicit) onto Ballast."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbos import DBOSConfig

    from ballast.app import Ballast


class DBOSProvider:
    """Register DBOS lifecycle on the :class:`Ballast` app.

    Pulls :class:`DBOSConfig` from ``ballast.settings.dbos`` by default,
    or accepts an explicit config for apps that need to override (e.g.
    when computing a tempfile SQLite URL at startup). When no
    ``database_url`` is configured anywhere the provider is a no-op —
    apps that don't need durable execution can still wire ``DBOSProvider()``
    safely.
    """

    def __init__(self, config: "DBOSConfig | None" = None) -> None:
        self._config = config

    def register(self, ballast: "Ballast") -> None:
        if self._config is not None:
            ballast._set_dbos(self._config)
            return
        # Build from settings.
        s = ballast.settings.dbos
        if not s.database_url:
            return  # nothing configured; DBOS not used
        from dbos import DBOSConfig

        config = DBOSConfig(
            name=s.app_name,
            system_database_url=s.database_url,
        )
        ballast._set_dbos(config)


__all__ = ["DBOSProvider"]

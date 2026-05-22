"""Built-in Ballast providers — registered onto the :class:`Ballast`
instance via ``.use(...)``.

Apps write their own :class:`Provider` instances for custom concerns
(metrics, multi-tenancy, etc.) and pass them to ``.use(...)`` alongside
the framework-built-ins.
"""
from ballast.providers.dbos import DBOSProvider
from ballast.providers.events import EventsProvider
from ballast.providers.observability import ObservabilityProvider
from ballast.providers.threads import ThreadsProvider

__all__ = [
    "DBOSProvider",
    "EventsProvider",
    "ObservabilityProvider",
    "ThreadsProvider",
]

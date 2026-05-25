"""``Ballast`` — the framework-agnostic application root.

Constructed once at app startup with a :class:`BallastSettings` instance,
then configured via ``.use(...)`` (one or more :class:`Provider`
instances). The final transport adapter (``.fastapi(...)``) returns the
FastAPI app; this is the only point where FastAPI is imported, so the
rest of the framework + apps can stay transport-agnostic.

Sits alongside the legacy :func:`ballast.create_app` helper — both
construct the same underlying :class:`Engine`. Providers
(:class:`DBOSProvider`, :class:`ThreadsProvider`, :class:`EventsProvider`,
:class:`ObservabilityProvider`) own one slice each so apps wire by
composition instead of a long kwargs list.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from fastapi import APIRouter, FastAPI

    from ballast.api.cors import CORSConfig
    from ballast.memory._scope import Scope
    from ballast.memory.episodic._facade import EpisodicMemory
    from ballast.memory.semantic._facade import SemanticMemory
    from ballast.persistence.approval_card import ApprovalCardRepository
    from ballast.settings import BallastSettings

try:  # Python 3.11+ has ``Self``; fall back for older bytecode caches.
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self  # type: ignore[assignment]


LifespanHook = Callable[["FastAPI"], Awaitable[None]]

_logger = logging.getLogger("ballast.app")


@runtime_checkable
class Provider(Protocol):
    """Configure one slice of the Ballast app at startup.

    Providers are plain Python objects that mutate the :class:`Ballast`
    instance during ``.use(...)``. The contract is a single
    ``register(ballast)`` call — implementations set repos / config on
    the Ballast object and may schedule lifespan hooks via
    ``ballast.add_on_startup`` / ``ballast.add_on_shutdown``.
    """

    def register(self, ballast: "Ballast") -> None: ...


class Ballast:
    """The Ballast app root. Constructed once per process.

    Builder pattern: ``.use(*providers)`` registers config and returns
    ``self`` for chaining. The terminal ``.fastapi(...)`` returns the
    FastAPI app and ALSO installs the process-wide :class:`Engine`
    singleton (mirroring :func:`ballast.create_app`).
    """

    def __init__(self, settings: "BallastSettings") -> None:
        from ballast.persistence.events.repository import (
            InMemoryEventLogRepository,
        )
        from ballast.persistence.thread.repository import (
            InMemoryThreadRepository,
        )
        from ballast.runtime.event_stream import InProcessEventStream

        self.settings = settings
        # Sensible defaults — apps override via providers.
        self._thread_repo: object = InMemoryThreadRepository()
        self._event_log: object = InMemoryEventLogRepository()
        self._event_stream: object = InProcessEventStream()
        self._observability_installed = False
        self._dbos_lifecycle = False
        self._dbos_config: object | None = None
        self._on_startup: list[LifespanHook] = []
        self._on_shutdown: list[LifespanHook] = []
        self._approval_repo: "ApprovalCardRepository | None" = None
        self._episodic_memory: "EpisodicMemory | None" = None
        self._memory: "EpisodicMemory | None" = None  # back-compat shadow
        self._semantic_memory: "SemanticMemory | None" = None

    def use(self, *providers: Provider) -> Self:
        """Register one or more third-party providers.

        Extension seam for app-defined slices (metrics, multi-tenancy,
        custom transports). Built-in framework wiring uses the fluent
        ``.with_*`` setters below instead — collapsing the four
        framework-built-in providers down to ordinary methods on
        :class:`Ballast` removes a layer of indirection without giving
        up the ``Provider`` Protocol for outside extension.
        """
        for provider in providers:
            provider.register(self)
        return self

    def add_on_startup(self, hook: LifespanHook) -> None:
        """Schedule a startup hook (called by ``.fastapi(...)`` lifespan)."""
        self._on_startup.append(hook)

    def add_on_shutdown(self, hook: LifespanHook) -> None:
        """Schedule a shutdown hook (reverse order on shutdown)."""
        self._on_shutdown.append(hook)

    # ── Fluent setters ───────────────────────────────────────────────

    def with_thread_repo(self, repo: object) -> Self:
        """Install the :class:`ThreadRepository` used by the
        :class:`Engine`."""
        self._thread_repo = repo
        return self

    def with_events(
        self, event_log: object, event_stream: object,
    ) -> Self:
        """Install the event-log repository + in-process event stream and
        connect the framework's default signal handlers.

        The defaults turn ``message_added`` / ``helper_thread_created``
        signal payloads into ``event_log.append`` +
        ``event_stream.publish`` calls. Apps that want different
        routing call ``Signal.disconnect`` on the relevant default and
        connect their own.
        """
        from ballast.events._default_handlers import (  # noqa: PLC0415
            connect_default_handlers,
        )

        self._event_log = event_log
        self._event_stream = event_stream
        connect_default_handlers()
        return self

    def with_dbos(self, config: "object | None" = None) -> Self:
        """Wire DBOS lifecycle onto the app.

        Pass an explicit :class:`DBOSConfig` for apps that compute one
        at startup (e.g. a tempfile SQLite URL). When omitted, builds
        one from :class:`BallastSettings.dbos`; when neither yields a
        ``database_url`` the call is a no-op so apps without durable
        execution can still call it safely.
        """
        if config is not None:
            self._dbos_config = config
            self._dbos_lifecycle = True
            return self
        s = self.settings.dbos
        if not getattr(s, "database_url", None):
            return self
        from dbos import DBOSConfig  # noqa: PLC0415

        self._dbos_config = DBOSConfig(
            name=s.app_name,
            system_database_url=s.database_url,
        )
        self._dbos_lifecycle = True
        return self

    def with_judge_defaults(
        self,
        model: str,
        *,
        model_settings: "object | None" = None,
    ) -> Self:
        """Set the process-wide default judge model + ``ModelSettings``.

        Every ``LLMJudge(model=None, model_settings=None, ...)`` in
        the app — including the ones inside :class:`JudgeAfterRun`
        capabilities — picks these up. Per-instance args still win.

        Thin convenience over :func:`set_default_judge_model`; lives
        on the builder so judge wiring stays consistent with the
        other ``.with_*`` slots instead of being an ad-hoc global
        call somewhere in module scope::

            ballast.Ballast(settings)
                .with_judge_defaults(
                    "openrouter:qwen/qwen3.6-plus",
                    model_settings=OpenRouterModelSettings(
                        temperature=0.0,
                        openrouter_reasoning={"effort": "none"},
                    ),
                )
                .with_thread_repo(thread_repo)
                .with_events(event_log, event_stream)
                ...

        Returns ``self`` for chaining. Idempotent — re-calling
        overwrites the previous default.
        """
        from ballast.capabilities.llm_judge import (  # noqa: PLC0415
            set_default_judge_model,
        )

        set_default_judge_model(model, model_settings=model_settings)
        return self

    def with_episodic_memory(
        self,
        memory: "EpisodicMemory",
        *,
        scope_builder: "Callable[[], Scope] | None" = None,
    ) -> Self:
        """Wire an EpisodicMemory facade + optional default scope-builder.

        Replaces the deprecated ``with_memory`` (kept as a backward-
        compatible alias for one release window).
        """
        self._episodic_memory = memory
        self._memory = memory  # back-compat shadow for Phase 1 consumers
        if scope_builder is not None:
            memory._default_scope_builder = scope_builder
        return self

    def with_semantic_memory(
        self,
        memory: "SemanticMemory",
    ) -> Self:
        """Wire a SemanticMemory facade for agent-pull tool exposure.

        Workflow code accesses semantic sources directly via their module
        singletons (e.g. ``from notes_app.memory.semantic_sources import
        notes_semantic``) — the facade is purely for agent tool collection.
        """
        self._semantic_memory = memory
        return self

    def with_memory(
        self,
        memory: "EpisodicMemory",
        *,
        scope_builder: "Callable[[], Scope] | None" = None,
    ) -> Self:
        """Deprecated — use ``with_episodic_memory(...)`` instead.

        Kept as a backward-compatible alias for one release window so
        existing Phase 1 wiring continues to work.
        """
        import warnings
        warnings.warn(
            "Ballast.with_memory is deprecated; use with_episodic_memory.",
            DeprecationWarning, stacklevel=2,
        )
        return self.with_episodic_memory(memory, scope_builder=scope_builder)

    def with_approval_repo(
        self, repo: "ApprovalCardRepository",
    ) -> Self:
        """Configure the approval-card repository (defaults to in-memory).

        Reassigns the module-level ``approval_card_repo`` singleton so
        tools / channels that do
        ``from ballast.persistence.approval_card import approval_card_repo``
        pick up the configured instance without explicit DI.

        Returns ``self`` for chaining. Idempotent — re-calling overwrites
        the previous repository.
        """
        import ballast.persistence.approval_card as _ac  # noqa: PLC0415

        _ac.approval_card_repo = repo
        self._approval_repo = repo
        return self

    def with_observability(
        self, config: "object | None" = None,
    ) -> Self:
        """Install :class:`ObservabilityConfig` (Logfire + instrumentation).

        Pass an explicit :class:`ObservabilityConfig` to override the
        knobs derived from :class:`BallastSettings.observability`.
        Idempotent — ``ObservabilityConfig.install`` is itself a no-op
        on repeat calls with the same config.
        """
        from ballast.observability.config import (  # noqa: PLC0415
            ObservabilityConfig,
        )

        resolved = config
        if resolved is None:
            s = self.settings.observability
            resolved = ObservabilityConfig(
                service_name=s.service_name,
                environment=s.environment,
                instrument_pydantic_ai=s.instrument_pydantic_ai,
                instrument_httpx=s.instrument_httpx,
                instrument_fastapi=s.instrument_fastapi,
            )
        resolved.install()  # type: ignore[attr-defined]
        self._observability_installed = True
        return self

    # ── Transport adapter ────────────────────────────────────────────

    def fastapi(
        self,
        *,
        cors: "CORSConfig | str | None" = None,
        routers: Sequence["APIRouter"] = (),
        health_checks: dict[str, Callable[[], Awaitable[bool]]] | None = None,
        **fastapi_kwargs: Any,
    ) -> "FastAPI":
        """Build the FastAPI app + install the :class:`Engine` singleton.

        ``cors`` accepts a :class:`CORSConfig` instance, the string
        ``"dev"`` (shortcut for ``CORSConfig.permissive_dev()``), or
        ``None`` (no CORS middleware).

        Any extra keyword arguments are forwarded verbatim to
        :class:`fastapi.FastAPI` — ``title``, ``version``, ``debug``,
        ``docs_url``, ``openapi_tags``, ``middleware``,
        ``exception_handlers``, etc. The only reserved kwarg is
        ``lifespan`` (Ballast owns it to run provider startup/shutdown
        hooks — use :meth:`add_on_startup` / :meth:`add_on_shutdown` to
        plug in your own).
        """
        if "lifespan" in fastapi_kwargs:
            raise TypeError(
                "Ballast.fastapi(...) reserves the 'lifespan' kwarg; use "
                "add_on_startup() / add_on_shutdown() to register hooks.",
            )
        from fastapi import FastAPI

        from ballast.api.cors import CORSConfig
        from ballast.api.approvals.router import approvals_router
        from ballast.api.dbos_router import dbos_router
        from ballast.api.error_middleware import install_error_handlers
        from ballast.api.health import build_health_router, health_router
        from ballast.api.threads import threads_router
        from ballast.durable import Durable
        from ballast.runtime.engine import Engine, _set_ballast

        # Engine construction + singleton install.
        engine = Engine(
            thread_repo=self._thread_repo,  # type: ignore[arg-type]
            event_log=self._event_log,  # type: ignore[arg-type]
            event_stream=self._event_stream,  # type: ignore[arg-type]
        )
        _set_ballast(engine)

        # DBOS lifespan hooks (only when :class:`DBOSProvider` was used).
        if self._dbos_lifecycle and self._dbos_config is not None:
            dbos_config = self._dbos_config

            async def _launch_dbos(_app: "FastAPI") -> None:
                Durable.init(dbos_config)  # type: ignore[arg-type]
                Durable.launch()

            async def _destroy_dbos(_app: "FastAPI") -> None:
                Durable.destroy(destroy_registry=False)

            self._on_startup.insert(0, _launch_dbos)
            self._on_shutdown.append(_destroy_dbos)

        # Auto-migrate hook — opt-in via ``settings.auto_migrate`` /
        # ``BALLAST_AUTO_MIGRATE=true``. Runs ``alembic upgrade head``
        # BEFORE all other startup hooks (incl. DBOS) so schemas exist
        # before anything touches the database. Skipped under pytest.
        settings_obj = self.settings

        async def _auto_migrate(_app: "FastAPI") -> None:
            import asyncio  # noqa: PLC0415
            import sys  # noqa: PLC0415

            if not getattr(settings_obj, "auto_migrate", False):
                return
            if "pytest" in sys.modules:
                return
            from alembic.config import main as alembic_main  # noqa: PLC0415

            from ballast._alembic import resolve_alembic_ini  # noqa: PLC0415

            ini = resolve_alembic_ini()

            # ``alembic upgrade head`` is sync, and app alembic ``env.py``
            # files often call ``asyncio.run`` for async URLs — both would
            # explode inside the running lifespan loop. Run in a worker
            # thread so the migration gets its own event-loop context.
            def _run_alembic() -> None:
                saved_argv = sys.argv
                try:
                    sys.argv = ["alembic", "-c", ini, "upgrade", "head"]
                    alembic_main()
                finally:
                    sys.argv = saved_argv

            await asyncio.to_thread(_run_alembic)
            _logger.info("alembic: upgraded to head")

        self._on_startup.insert(0, _auto_migrate)

        startup = list(self._on_startup)
        shutdown = list(self._on_shutdown)

        @asynccontextmanager
        async def _lifespan(_app: "FastAPI") -> AsyncIterator[None]:
            for hook in startup:
                try:
                    await hook(_app)
                except Exception:
                    _logger.exception(
                        "startup hook %r raised; aborting boot",
                        getattr(hook, "__qualname__", repr(hook)),
                    )
                    raise
            try:
                yield
            finally:
                for hook in reversed(shutdown):
                    try:
                        await hook(_app)
                    except Exception:
                        _logger.exception(
                            "shutdown hook %r raised; continuing",
                            getattr(hook, "__qualname__", repr(hook)),
                        )

        app = FastAPI(lifespan=_lifespan, **fastapi_kwargs)
        app.state.engine = engine

        # Resolve CORS shortcut.
        cors_config: "CORSConfig | None"
        if isinstance(cors, str):
            if cors == "dev":
                cors_config = CORSConfig.permissive_dev()
            else:
                raise ValueError(f"Unknown cors shortcut {cors!r}")
        else:
            cors_config = cors

        # Mount built-in routers.
        if health_checks is not None:
            app.include_router(build_health_router(checks=health_checks))
        else:
            app.include_router(health_router)
        app.include_router(threads_router)
        app.include_router(dbos_router)
        app.include_router(approvals_router)

        # Extra routers from the app.
        for r in routers:
            app.include_router(r)

        # Structured-error handlers FIRST so CORSMiddleware ends up
        # outermost. Starlette's ``add_middleware`` prepends — last added
        # = outermost. If CORS is inner and an exception turns into an
        # error response via BallastErrorMiddleware (outer), CORS never
        # sees the response and headers don't get attached, and the
        # browser masks the real 500 as a CORS failure.
        install_error_handlers(app)

        # CORS — added last so it wraps everything, including error
        # responses produced by BallastErrorMiddleware.
        if cors_config is not None:
            from fastapi.middleware.cors import CORSMiddleware

            app.add_middleware(
                CORSMiddleware,
                allow_origins=list(cors_config.allow_origins),
                allow_methods=list(cors_config.allow_methods),
                allow_headers=list(cors_config.allow_headers),
                allow_credentials=cors_config.allow_credentials,
                expose_headers=list(cors_config.expose_headers),
                max_age=cors_config.max_age,
            )

        return app


__all__ = ["Ballast", "LifespanHook", "Provider"]

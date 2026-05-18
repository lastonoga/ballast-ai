from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic_ai_stateflow.runtime.container import Container


@runtime_checkable
class ServiceProvider(Protocol):
    """Single-phase provider (per spec 4A.0.13).

    Replaces the original two-phase register/boot ceremony (which was
    enterprise overhead for 8 manually-ordered providers, per code-review).

    Providers are registered in user-declared order in Engine constructor.
    If provider B depends on provider A's bindings, B must come after A
    in the list.

    Engine runs post-registration invariants (Alembic check, Tool coverage,
    etc) AFTER all providers have registered.
    """

    async def register(self, container: Container) -> None:
        """Bind everything this provider owns + initialise as needed.

        Free to perform async I/O (e.g. preheating caches), but must not
        depend on other providers' instances mid-registration — only on
        their bindings (resolved lazily on `container.get`).
        """

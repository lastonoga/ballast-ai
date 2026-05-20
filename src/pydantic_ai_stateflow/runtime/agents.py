"""``StateflowAgent`` ABC + process-wide registry.

A ``StateflowAgent`` subclass binds together everything the framework needs
to drive a thread end-to-end:

  - ``name`` (ClassVar)         — registry key. Stored verbatim as
    ``Thread.agent``.
  - ``metadata_model`` (ClassVar) — optional Pydantic model used to
    validate ``Thread.metadata`` at create-time. ``None`` ⇒ free-form.
  - ``build_agent()``           — constructs the underlying pydantic-ai
    ``Agent`` (tools, system prompt, output type). Called once and
    cached on the instance via ``self.agent``.
  - ``build_deps(...)``         — mints per-request deps for the agent
    run. Receives the thread, tenant id, and the just-arrived user
    message.
  - ``model_settings()``        — optional ``ModelSettings`` forwarded
    on every run (temperature, provider config, etc.).

Apps subclass it, instantiate it with whatever app-level dependencies
they need (repos, settings, embedders), and call ``register_agent(instance)``
during startup. The streaming router resolves the right instance from
``thread.agent`` per request — apps never wire the pydantic-ai ``Agent``
into the framework manually.

The registry is process-global and unsynchronized; concurrent
registration during startup is fine because Python imports serialize
through the GIL, and post-boot the registry is read-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from uuid import UUID

    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.settings import ModelSettings

    from pydantic_ai_stateflow.persistence.thread.domain import Thread


class StateflowAgent(ABC):
    """Framework-owned agent abstraction. One subclass per ``Thread.agent`` key.

    Subclasses MUST set ``name`` (ClassVar). Everything else has a sane
    default that callers can override.
    """

    name: ClassVar[str]
    """Registry key — written verbatim into ``Thread.agent``."""

    metadata_model: ClassVar[type[BaseModel] | None] = None
    """Optional Pydantic model validating ``Thread.metadata`` on create.

    When set, ``validate_thread_metadata(cls, raw)`` round-trips ``raw``
    through ``metadata_model.model_validate(...).model_dump(mode="json")``.
    When ``None``, metadata passes through unchanged.
    """

    @abstractmethod
    def build_agent(self) -> Agent[Any, Any]:
        """Return the pydantic-ai ``Agent`` driving threads of this kind.

        Called once per process and cached on ``self.agent``. Construct
        the model, register tools, set the system prompt, declare the
        output type — anything stable across all threads of this agent.

        Per-request variation belongs in ``build_deps``.
        """

    @abstractmethod
    async def build_deps(
        self,
        *,
        thread: Thread,
        tenant_id: UUID,
        message: ModelMessage | None,
    ) -> Any:
        """Mint per-request deps for the agent's ``deps_type``.

        ``thread`` carries metadata; ``message`` is the just-arrived
        user turn (or ``None`` for auto-resend after approval). Return
        whatever the underlying ``Agent.deps_type`` expects.
        """

    def model_settings(self) -> ModelSettings | None:
        """Forwarded to every agent run. Defaults to ``None``."""
        return None

    @cached_property
    def agent(self) -> Agent[Any, Any]:
        """Lazy-cached pydantic-ai ``Agent``. First access calls ``build_agent``.

        Caching here serves two purposes:

        1. Defers heavy construction (API key lookups, tool registration,
           model client init) until the first request actually arrives —
           tests and admin endpoints that never stream can boot the app
           without the env vars.
        2. Keeps the ``Agent`` stable across requests; pydantic-ai
           registers tools at construction time and they don't need to
           re-register per call.
        """
        return self.build_agent()


# ── Registry ─────────────────────────────────────────────────────────────────

_registry: dict[str, StateflowAgent] = {}


def register_agent(agent: StateflowAgent) -> None:
    """Register an agent instance under its ``type(agent).name``.

    Idempotent: re-registering the same name overwrites — useful for
    tests that swap implementations between cases. Apps register exactly
    once per agent class at startup.
    """
    cls = type(agent)
    if not hasattr(cls, "name") or not isinstance(cls.name, str) or not cls.name:
        raise ValueError(
            f"{cls.__name__}.name must be a non-empty ClassVar[str]",
        )
    _registry[cls.name] = agent


def get_agent(name: str) -> StateflowAgent:
    """Return the registered agent instance for ``name``.

    Raises ``KeyError`` with a helpful message listing known names when
    the lookup fails — most often this means the app forgot to call
    ``register_agent(...)`` during startup or wrote a typo in
    ``Thread.agent``.
    """
    try:
        return _registry[name]
    except KeyError as exc:
        known = sorted(_registry)
        raise KeyError(
            f"No StateflowAgent registered for name {name!r}. "
            f"Known agents: {known}. "
            f"Did you call register_agent(...) at app startup?",
        ) from exc


def list_agents() -> list[StateflowAgent]:
    """Return all registered agents (snapshot, stable iteration order)."""
    return [_registry[name] for name in sorted(_registry)]


def clear_agent_registry() -> None:
    """Drop every registered agent. Tests use this in fixtures; apps never call it."""
    _registry.clear()


# ── Selector + metadata validation ──────────────────────────────────────────

AgentRef = "type[StateflowAgent] | StateflowAgent | str"
"""What ``validate_thread_metadata`` and friends accept as an agent
selector. The class itself, an instance, or the registered string name."""


def _resolve_agent_name(ref: Any) -> str:
    """Coerce ``AgentRef`` → registered name string.

    Accepts the string itself, a ``StateflowAgent`` subclass, or an
    instance of one. Anything else raises ``TypeError``.
    """
    if isinstance(ref, str):
        return ref
    if isinstance(ref, type) and issubclass(ref, StateflowAgent):
        return ref.name
    if isinstance(ref, StateflowAgent):
        return type(ref).name
    raise TypeError(
        f"AgentRef must be str | type[StateflowAgent] | StateflowAgent, "
        f"got {type(ref).__name__}",
    )


def validate_thread_metadata(
    ref: Any, raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate ``raw`` against the agent's ``metadata_model``.

    - ``raw`` of ``None`` is normalized to ``{}``.
    - When ``metadata_model`` is ``None`` the dict passes through
      (after a defensive copy) without schema checks.
    - Otherwise: ``model_validate(raw)`` → ``model_dump(mode="json")``
      so the persisted shape is canonical JSON.

    Raises ``KeyError`` if no agent is registered for ``ref``, or
    ``ValidationError`` (from pydantic) if the metadata is invalid.
    """
    payload: dict[str, Any] = dict(raw or {})
    name = _resolve_agent_name(ref)
    instance = get_agent(name)
    model = type(instance).metadata_model
    if model is None:
        return payload
    return model.model_validate(payload).model_dump(mode="json")


__all__ = [
    "AgentRef",
    "StateflowAgent",
    "clear_agent_registry",
    "get_agent",
    "list_agents",
    "register_agent",
    "validate_thread_metadata",
]

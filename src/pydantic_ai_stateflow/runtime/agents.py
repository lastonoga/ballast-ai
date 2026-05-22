"""``StateflowAgent`` ABC + process-wide registry.

A ``StateflowAgent`` subclass binds together everything the framework needs
to drive a thread end-to-end:

  - ``name`` (ClassVar)         — registry key. Stored verbatim as
    ``Thread.agent``.
  - ``metadata_model`` (ClassVar) — optional Pydantic model used to
    validate ``Thread.metadata`` at create-time. ``None`` ⇒ free-form.
  - ``build_agent()``           — constructs the underlying pydantic-ai
    ``Agent`` (model, system prompt, output type). Tool registration
    and grounded ``Ref[T]/Selector`` hooks are applied automatically by
    the framework, AFTER ``build_agent`` returns — subclasses don't call
    ``agent.tool(...)`` or ``register_grounded_tools(...)`` themselves.
  - ``build_deps(...)``         — mints per-request deps for the agent
    run. Receives the thread, tenant id, and the just-arrived user
    message.
  - ``model_settings()``        — optional ``ModelSettings`` forwarded
    on every run (temperature, provider config, etc.).

**Tools as class-level declarations.** Tool functions are registered on
the subclass via the ``@SomeAgent.tool`` decorator at module load (one
file per agent, no ``register_*()`` wrapper). The framework collects them
during ``StateflowAgent.agent`` (cached_property) construction and calls
``agent.tool(...)`` / ``agent.tool_plain(...)`` based on whether the
function takes a ``RunContext`` first argument. Tools defined on parent
classes are inherited by subclasses.

**Grounded ``Ref[T]/Selector`` hooks are implicit.** Any tool parameter
annotated with ``Annotated[Ref[T], Selector(...)]`` automatically gets a
per-run ``prepare`` hook that narrows its JSON Schema to a closed enum.
No explicit ``register_grounded_tools(agent)`` call needed.

Apps subclass ``StateflowAgent`` to get the tool / system-prompt
helpers; instances are constructed and used directly. The framework
no longer maintains a class registry — ``Thread.agent`` is an opaque
app-owned string and apps resolve string→instance themselves in
their own routes.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, get_origin, get_type_hints

from pydantic_ai import RunContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.settings import ModelSettings

    from pydantic_ai_stateflow.persistence.thread.domain import Thread


@dataclass
class _ToolEntry:
    """One ``@SomeAgent.tool`` registration."""

    fn: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SystemPromptEntry:
    """One ``@SomeAgent.system_prompt`` registration."""

    fn: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)


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

    # Per-subclass tool registry. ``__init_subclass__`` gives each
    # subclass its OWN list so parent registrations don't leak across
    # sibling subclasses. Tools defined on a parent ARE inherited (the
    # ``agent`` cached_property walks ``__mro__`` to collect them).
    _tools: ClassVar[list[_ToolEntry]] = []
    _system_prompts: ClassVar[list[_SystemPromptEntry]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._tools = []
        cls._system_prompts = []

    # ── tool decorator (class-level) ─────────────────────────────────────────

    @classmethod
    def tool(
        cls,
        func: Callable[..., Any] | None = None,
        /,
        **tool_kwargs: Any,
    ) -> Any:
        """Decorator to register a function as a tool on this agent class.

        Two usage forms — same as pydantic-ai's ``@agent.tool``::

            @NotesAgent.tool
            async def create_note(ctx: RunContext[NoteToolDeps], ...): ...

            @NotesAgent.tool(requires_approval=True)
            async def delete_note(ctx: RunContext[NoteToolDeps], ...): ...

        Whether the function gets ``RunContext`` as its first arg is
        detected automatically via ``get_type_hints`` — ctx-bearing
        functions route to ``Agent.tool`` and plain ones to
        ``Agent.tool_plain``.

        ``tool_kwargs`` are forwarded verbatim to the pydantic-ai
        decorator: ``name``, ``description``, ``retries``, ``prepare``,
        ``docstring_format``, ``requires_approval``, ``metadata``,
        ``sequential``, ``timeout``, ``strict`` are all supported.

        .. note::
           Tools run **inline** inside ``StateflowDurableAgent``'s
           workflow context — they are NOT wrapped in ``@DBOS.step``.
           ``DBOSAgent`` (the upstream wrapper we delegate to) only
           step-wraps model requests + MCP toolsets, not regular
           ``@agent.tool`` functions. Consequence: on workflow replay
           (crash recovery) tool side effects re-fire, and tools
           don't appear in the DBOS step log. Apps that need
           idempotency must implement it themselves (e.g. INSERT ON
           CONFLICT DO NOTHING) or wrap critical paths in
           ``@DBOS.step`` manually.
        """

        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            cls._tools.append(_ToolEntry(
                fn=fn, kwargs=dict(tool_kwargs),
            ))
            return fn

        # Bare ``@NotesAgent.tool`` (no args) → ``func`` is the function.
        # ``@NotesAgent.tool(...)`` → ``func`` is None, return ``register``.
        if func is not None:
            return register(func)
        return register

    @classmethod
    def system_prompt(
        cls,
        func: Callable[..., Any] | None = None,
        /,
        **prompt_kwargs: Any,
    ) -> Any:
        """Decorator to register a system-prompt callback on this agent class.

        Mirrors pydantic-ai's ``@agent.system_prompt`` — the function
        runs per agent run and its returned string is appended to the
        base system prompt. Sync or async, with or without
        ``RunContext`` as the first argument::

            @TodoApprovalAgent.system_prompt
            def _show_context(ctx: RunContext[TodoApprovalDeps]) -> str:
                return ctx.deps.metadata.to_system_prompt()

            @TodoApprovalAgent.system_prompt(dynamic=True)
            async def _async_prompt(ctx: RunContext[TodoApprovalDeps]) -> str:
                ...

        ``prompt_kwargs`` are forwarded verbatim to pydantic-ai's
        ``@agent.system_prompt`` (e.g. ``dynamic=True``). Inheritance
        works the same way as for tools — parent class system prompts
        are collected via MRO walk in the ``agent`` cached property.
        """

        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            cls._system_prompts.append(
                _SystemPromptEntry(fn=fn, kwargs=dict(prompt_kwargs)),
            )
            return fn

        if func is not None:
            return register(func)
        return register

    # ── subclass hooks ───────────────────────────────────────────────────────

    @abstractmethod
    def build_agent(self) -> Agent[Any, Any]:
        """Return the pydantic-ai ``Agent`` driving threads of this kind.

        Construct the model, set the system prompt, declare the output
        type — anything stable across all threads of this agent. Do NOT
        call ``agent.tool(...)`` here; the framework registers tools
        declared via ``@SomeAgent.tool`` automatically. Do NOT call
        ``register_grounded_tools(agent)`` either; ``Ref[T]/Selector``
        hooks are applied automatically.

        Per-request variation belongs in ``build_deps``.
        """

    @abstractmethod
    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> Any:
        """Mint per-request deps for the agent's ``deps_type``.

        ``thread`` carries metadata (any per-request scope the app needs
        — tenant_id, user_id, workspace_id — lives there). ``message``
        is the just-arrived user turn (or ``None`` for auto-resend
        after approval).
        """

    def model_settings(self) -> ModelSettings | None:
        """Forwarded to every agent run. Defaults to ``None``."""
        return None

    # ── lazy Agent construction ──────────────────────────────────────────────

    @cached_property
    def agent(self) -> Agent[Any, Any]:
        """Lazy-cached pydantic-ai ``Agent``.

        On first access:

        1. ``build_agent()`` constructs the bare Agent.
        2. Tools declared with ``@SomeAgent.tool`` on this class and all
           its ancestors are registered verbatim (no wrapping —
           ``DBOSAgent`` does NOT step-wrap ``@agent.tool`` functions,
           so we can't either without losing the ``DBOS.step`` opt-in
           semantics).
        3. ``register_grounded_tools(agent)`` installs ``prepare`` hooks
           on any tool parameter annotated with
           ``Annotated[Ref[T], Selector(...)]``.

        Caching defers heavy construction (API key lookups, tool
        registration, model client init) until the first request
        actually arrives — tests and admin endpoints that never stream
        can boot the app without env vars.
        """
        a = self.build_agent()
        for entry in _collect_inherited_tools(type(self)):
            if _takes_run_context(entry.fn):
                a.tool(**entry.kwargs)(entry.fn)
            else:
                a.tool_plain(**entry.kwargs)(entry.fn)
        for sp in _collect_inherited_system_prompts(type(self)):
            a.system_prompt(**sp.kwargs)(sp.fn)
        # Auto-apply grounded ``Ref[T]/Selector`` prepare hooks. Importing
        # here (not at module top) keeps ``runtime.agents`` free of a
        # hard dep on the ``grounded`` package at import time, which
        # matters because ``grounded`` itself imports from
        # ``pydantic_ai_stateflow.runtime`` transitively.
        from pydantic_ai_stateflow.grounded import (  # noqa: PLC0415
            register_grounded_tools,
        )
        register_grounded_tools(a)
        return a


def _collect_inherited_tools(cls: type[StateflowAgent]) -> list[_ToolEntry]:
    """Walk MRO base-to-derived, gathering ``_tools`` from each class.

    Each subclass owns its own ``_tools`` list (set up by
    ``__init_subclass__``); this function aggregates them so that
    subclass tools extend (don't replace) parent tools. Same-name
    overrides on a subclass take precedence — the parent's tool is
    dropped when a later class registers a tool with the same Python
    function name.
    """
    collected: dict[str, _ToolEntry] = {}
    for klass in reversed(cls.__mro__):
        own = klass.__dict__.get("_tools")
        if not own:
            continue
        for entry in own:
            collected[entry.fn.__name__] = entry
    return list(collected.values())


def _collect_inherited_system_prompts(
    cls: type[StateflowAgent],
) -> list[_SystemPromptEntry]:
    """Walk MRO base-to-derived, gathering ``_system_prompts`` from each class.

    Unlike tools, system prompts are ADDITIVE — pydantic-ai appends each
    callback's return value to the base prompt, so duplicate function
    names don't override. Order is base-to-derived to keep parent
    context above subclass refinements in the final prompt.
    """
    collected: list[_SystemPromptEntry] = []
    for klass in reversed(cls.__mro__):
        own = klass.__dict__.get("_system_prompts")
        if not own:
            continue
        collected.extend(own)
    return collected


def _takes_run_context(fn: Callable[..., Any]) -> bool:
    """Return True iff ``fn``'s first positional parameter is a ``RunContext``.

    pydantic-ai's ``@agent.tool`` requires the first arg to be a
    ``RunContext``; ``@agent.tool_plain`` is for ctx-less tools. We
    detect via ``get_type_hints`` so ``from __future__ import annotations``
    still works.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return False
    first = params[0]
    try:
        hints = get_type_hints(fn)
    except Exception:
        return False
    anno = hints.get(first.name)
    if anno is None:
        return False
    origin = get_origin(anno) or anno
    return origin is RunContext


# ── Metadata validation ──────────────────────────────────────────────────
#
# ``Thread.agent`` is an opaque app-owned string; the framework doesn't
# resolve names → classes. Apps that want metadata validation pass the
# class (or an instance) to ``validate_thread_metadata`` directly.

AgentRef = "type[StateflowAgent] | StateflowAgent"
"""What ``validate_thread_metadata`` accepts as an agent selector:
the class itself or an instance."""


def validate_thread_metadata(
    ref: Any, raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate ``raw`` against the agent's ``metadata_model``.

    - ``raw`` of ``None`` is normalized to ``{}``.
    - When ``metadata_model`` is ``None`` the dict passes through
      (after a defensive copy) without schema checks.
    - Otherwise: ``model_validate(raw)`` → ``model_dump(mode="json")``
      so the persisted shape is canonical JSON.

    ``ref`` may be a ``StateflowAgent`` class or instance.

    Raises ``ValidationError`` (from pydantic) if the metadata is invalid.
    """
    payload: dict[str, Any] = dict(raw or {})
    if isinstance(ref, type) and issubclass(ref, StateflowAgent):
        cls = ref
    elif isinstance(ref, StateflowAgent):
        cls = type(ref)
    else:
        raise TypeError(
            f"AgentRef must be type[StateflowAgent] | StateflowAgent, "
            f"got {type(ref).__name__}",
        )
    model = cls.metadata_model
    if model is None:
        return payload
    return model.model_validate(payload).model_dump(mode="json")


__all__ = [
    "AgentRef",
    "StateflowAgent",
    "validate_thread_metadata",
]

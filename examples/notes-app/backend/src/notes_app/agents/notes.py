"""OpenRouter-backed ``StateflowAgent`` for the notes app.

One file = one agent. ``NotesAgent`` is the framework's per-thread agent
abstraction (see
``pydantic_ai_stateflow.runtime.agents.StateflowAgent``); the registry
binds ``Thread.agent == "notes"`` to this class.

Tools are declared inline with ``@NotesAgent.tool`` decorators at module
load — the framework registers them on the underlying pydantic-ai
``Agent`` automatically when ``self.agent`` is first accessed. Grounded
``Annotated[Ref[Note], Selector(...)]`` parameters get per-run ``prepare``
hooks installed automatically too; no explicit
``register_grounded_tools(...)`` call needed.

Output shape decision (iter 3 round 2, still relevant):
  We use ``output_type=[str, DeferredToolRequests]`` — NOT a structured
  ``BaseModel`` envelope. Reasons:

  - ``ToolOutput`` (pydantic-ai's default for ``BaseModel``) forces
    ``tool_choice="required"`` to drive the synthetic ``final_result``
    tool. OpenRouter's Qwen 3.6 endpoints reject that value.
  - ``NativeOutput`` accepts ``response_format: json_schema`` but the
    model then returns the JSON directly WITHOUT calling real tools.
  - ``PromptedOutput`` works but pollutes the streamed text with JSON.

  Plain ``str`` (plus ``DeferredToolRequests`` for the approval branch)
  sidesteps all three.

Note on annotations: we intentionally do NOT use
``from __future__ import annotations`` here. pydantic-ai's tool-
registration introspects parameter types via ``get_type_hints()`` at
decoration time, and postponed-evaluation annotations have bitten this
codebase before (see ``project_pydantic_ai_api_quirks.md``).
"""

import re
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow.capabilities import (
    BudgetGuard,
    PIIGuard,
    RegexDetector,
    StateflowCapability,
)
from pydantic_ai_stateflow.errors import MissingDependencyError
from pydantic_ai_stateflow.grounded import Ref, Selector
from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.runtime import StateflowDurableAgent

from notes_app.agents.todo_approval import NotesTodoApprovalAgent
from notes_app.models.note import Note
from notes_app.models.todo_approval import TodoApprovalContext
from notes_app.repositories.note import NoteRepository
from notes_app.settings import get_notes_settings

DEFAULT_MODEL = "qwen/qwen3.6-plus"
DEFAULT_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are the assistant inside a personal notes app. "
    "You have tools to create, list, search, edit, and delete notes on "
    "the user's behalf. "
    "When the user asks you to create / find / change / remove a note, "
    "USE THE TOOLS to actually do it — do not just describe what you "
    "would do. After running the tools, briefly confirm what happened "
    "(e.g. 'Saved your note titled \"X\"'). "
    "If the user is chatting and not asking for a note action, just "
    "reply conversationally."
)


# Naive but useful PII patterns for the demo — apps with real privacy
# constraints would replace these with NER or a vetted policy library.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{8,}\d")


def default_notes_capabilities() -> list[StateflowCapability]:
    """Capabilities applied to every ``NotesAgent`` run.

    - ``BudgetGuard`` bounds iteration count + token spend per run so a
      runaway tool-call loop, a chatty model, or a hostile prompt can't
      hang the request or burn unbounded budget.
    - ``PIIGuard`` regex-scrubs emails / phone numbers from the model's
      text replies BEFORE they're persisted to the thread, streamed to
      the frontend, or shipped to logs. The detector is pluggable —
      apps that need DB-grounded leak detection (e.g. "is this email
      one of OUR users in another thread?") swap ``RegexDetector`` for
      a custom ``PIIDetector`` that hits their user repo.
    """
    return [
        BudgetGuard(
            max_iterations=20,
            max_input_tokens=50_000,
            max_output_tokens=8_000,
        ),
        PIIGuard(
            detector=RegexDetector(
                patterns_by_category={
                    "email": [_EMAIL_RE],
                    "phone": [_PHONE_RE],
                },
            ),
        ),
    ]


@dataclass
class NoteToolDeps:
    """Per-request dependencies for the note tools.

    ``parent_thread_id`` is only used by ``propose_todo`` to spawn the
    durable approval workflow — the simpler note tools ignore it. It
    may be ``None`` for tests that only exercise the non-HITL tools.
    The note repository is reached via direct import of
    ``notes_app.repositories.note.notes_repo`` (module-level
    singleton); ``repo`` is kept on the deps so tools can pass a
    per-test override (constructed manually in unit tests).

    Framework infra (thread repo / event log / event stream) is reached
    by ``propose_todo`` via ``sf.get_engine()`` — no per-call context
    is threaded through ``NoteToolDeps`` anymore.
    """

    repo: NoteRepository
    parent_thread_id: UUID | None = None


class NotesAgent(StateflowDurableAgent):
    """Personal-notes durable agent.

    Extends ``StateflowDurableAgent`` so the run loop is a
    ``@DBOS.workflow`` — survives SSE disconnects, process restarts,
    and resumable via Last-Event-ID. Tools / system_prompt / metadata
    semantics are identical to ``StateflowAgent``; the only difference
    is the constructor (which now needs ``thread_repo`` + ``event_log``
    + ``event_stream`` for the durable infrastructure) and the run
    loop (which the streaming router dispatches into a workflow).
    """

    name = "notes"
    metadata_model = None  # no per-thread settings yet

    def build_agent(self) -> Agent[NoteToolDeps, Any]:
        settings = get_notes_settings()
        resolved_model = settings.openrouter_default_model or DEFAULT_MODEL
        resolved_key = (
            settings.openrouter_api_key.get_secret_value()
            if settings.openrouter_api_key else None
        )
        if not resolved_key:
            raise MissingDependencyError(
                "OpenRouter API key required to build NotesAgent",
                hint="Set NOTES_APP_OPENROUTER_API_KEY (or legacy OPENROUTER_API_KEY) env var",
                context={"setting": "notes_app.openrouter_api_key"},
            )

        model = OpenRouterModel(
            resolved_model,
            provider=OpenRouterProvider(api_key=resolved_key),
        )

        # ``output_type=[str, DeferredToolRequests]`` opts into pydantic-ai's
        # deferred-tools branch: when the model calls a tool marked
        # ``requires_approval=True`` (e.g. ``delete_note``) the agent
        # pauses and yields a ``DeferredToolRequests`` instead of
        # looping forever over an unresolved tool call.
        return Agent(
            model=model,
            output_type=[str, DeferredToolRequests],
            deps_type=NoteToolDeps,
            system_prompt=SYSTEM_PROMPT,
            capabilities=default_notes_capabilities(),
        )

    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> NoteToolDeps:
        del message
        # Direct import of the module-level singleton — apps that need
        # a different repo (e.g. tests) monkeypatch
        # ``notes_app.repositories.note.notes_repo`` before this runs.
        from notes_app.repositories.note import notes_repo

        return NoteToolDeps(
            repo=notes_repo,
            parent_thread_id=thread.id,
        )

    def model_settings(self) -> OpenRouterModelSettings:
        """Hardcoded OpenRouter settings for the notes-app demo.

        The Alibaba-upstream ``content: null`` rejection (see
        ``KNOWN_BUGS.md`` B9) is fixed at the framework layer via
        ``AssistantMessageNormalizer`` — apps don't need to route
        around it here.
        """
        return OpenRouterModelSettings(
            temperature=DEFAULT_TEMPERATURE,
            openrouter_reasoning={"effort": "none"},
            openrouter_usage={"include": True},
        )


# ── Tools ────────────────────────────────────────────────────────────────────
#
# Declared at module load via ``@NotesAgent.tool``. The framework
# registers them on the underlying pydantic-ai ``Agent`` the first time
# ``NotesAgent.agent`` is accessed, and auto-installs grounded
# ``prepare`` hooks for any ``Annotated[Ref[T], Selector(...)]`` params.


@NotesAgent.tool
async def create_note(
    ctx: RunContext[NoteToolDeps], title: str, body: str,
) -> Note:
    """Create a new note with the given title and body.

    Returns the saved note (including its ``id``, which you should use
    for any follow-up edit/delete in the same turn).

    Persisted by default (``StateflowDurableAgent`` wraps in @DBOS.step) —
    crash recovery returns the memoized note instead of creating a
    duplicate.
    """
    return await ctx.deps.repo.create(title=title, body=body)


@NotesAgent.tool
async def list_notes(
    ctx: RunContext[NoteToolDeps], limit: int = 20,
) -> list[Note]:
    """List the most recent notes for the current user, newest first.

    Use this when the user asks "show me my notes" or wants an
    overview. Returns at most ``limit`` notes (default 20).
    """
    return await ctx.deps.repo.list_(limit=limit)


@NotesAgent.tool
async def search_notes(
    ctx: RunContext[NoteToolDeps], query: str, limit: int = 20,
) -> list[Note]:
    """Search the user's notes by case-insensitive substring on title+body.

    Returns matching notes newest-first, at most ``limit``. Use this
    when the user references a note by topic or keyword rather than id.
    """
    return await ctx.deps.repo.search(query, limit=limit)


@NotesAgent.tool
async def edit_note(
    ctx: RunContext[NoteToolDeps],
    note_id: Annotated[
        Ref[Note],
        Selector(lambda c: c.deps.repo.list_(limit=1000)),
    ],
    title: str | None = None,
    body: str | None = None,
) -> Note:
    """Edit an existing note. Pass only the fields you want to change.

    ``note_id`` is constrained at the schema level (via Selector) to
    the set of notes that currently exist for this user — you cannot
    fabricate one. Returns the updated note.
    """
    nid = note_id.id if isinstance(note_id, Ref) else note_id
    return await ctx.deps.repo.update(nid, title=title, body=body)


@NotesAgent.tool
async def propose_todo(
    ctx: RunContext[NoteToolDeps], title: str, body: str,
) -> str:
    """Open a confirmation thread for a todo and return immediately.

    Tools run inline in the agent's DBOS workflow context (StateflowDurableAgent
    does NOT step-wrap them — DBOSAgent only step-wraps model requests
    and MCP), so calling ``DBOS.start_workflow_async`` here works
    directly: the spawned helper workflow is itself durable on its own.

    Use this INSTEAD of ``create_note`` when the user asks to create a
    TODO specifically. The flow is **fire-and-forget + durable**:

    1. A new thread is spawned bound to the ``todo_approval`` agent.
    2. A DBOS workflow (``TodoApprovalFlow``) is launched in the
       background — it blocks on the helper's HITL response, then saves
       the note (or skips on reject) and posts a notification message
       back to this thread.
    3. This tool returns IMMEDIATELY. The user sees "I opened a side
       conversation" right away, and when the helper agent resolves —
       even minutes or hours later, even if the user closed and
       reopened the app — the saved-todo notification will appear on
       this thread.

    Both ``title`` and ``body`` are required and free-form.
    """
    # Direct import of the module-level singleton.
    from notes_app.workflows.todo_approval import todo_flow

    if ctx.deps.parent_thread_id is None:
        raise RuntimeError(
            "propose_todo requires parent_thread_id on NoteToolDeps "
            "— was NotesAgent.build_deps run without a thread?",
        )

    context = TodoApprovalContext(
        proposed_title=title,
        proposed_body=body,
        parent_thread_id=ctx.deps.parent_thread_id,
    )
    await todo_flow.open(
        helper_agent=NotesTodoApprovalAgent,
        context=context,
        opening_message=context.to_opening_message(),
        notify_parent_thread_id=ctx.deps.parent_thread_id,
    )
    return (
        "I opened a confirmation thread to review this todo. "
        "Once you approve (or modify / reject) it there, I'll save "
        "it on this thread."
    )


@NotesAgent.tool(requires_approval=True)
async def delete_note(
    ctx: RunContext[NoteToolDeps],
    note_id: Annotated[
        Ref[Note],
        Selector(lambda c: c.deps.repo.list_(limit=1000)),
    ],
) -> str:
    """Delete the note with the given id.

    REQUIRES USER APPROVAL — destructive, irreversible. The model
    proposes the call; pydantic-ai pauses the run and emits a deferred
    ``approval-requested`` part. The frontend renders an approve/cancel
    card; once the user clicks, the response round-trips back through
    ``VercelAIAdapter.deferred_tool_results`` and this body executes
    (or denial is fed back to the model).

    ``note_id`` is constrained at the schema level (via Selector) to
    the set of notes that currently exist for this user. Idempotent —
    safe to call twice. Returns a short confirmation.
    """
    nid = note_id.id if isinstance(note_id, Ref) else note_id
    await ctx.deps.repo.delete(nid)
    return f"deleted {nid}"


# ── Module-level singleton ──────────────────────────────────────────────
# App-specific durable agent. Imported directly by ``main.py``'s
# dispatch table; per-request infra is bound on the instance via
# ``enqueue_run`` / ``enqueue_approval_resume`` from the supplied
# ``RunContext``.

notes_agent: NotesAgent = NotesAgent()

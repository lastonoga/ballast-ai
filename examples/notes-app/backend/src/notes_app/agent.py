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

import os
import re
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID, uuid4

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
from pydantic_ai_stateflow.grounded import Ref, Selector
from pydantic_ai_stateflow.patterns.hitl import (
    ApprovedResponse,
    HITLGate,
    HITLPrompt,
    ModifiedResponse,
)
from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
from pydantic_ai_stateflow.runtime import StateflowAgent

from notes_app.notes.domain import Note
from notes_app.notes.repository import NoteRepository

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

    ``hitl_gate``, ``thread_repo``, and ``parent_thread_id`` are only used
    by the HITL-gated ``propose_todo`` tool — the simpler note tools
    ignore them. They may be ``None`` for tests that only exercise the
    non-HITL tools.
    """

    repo: NoteRepository
    hitl_gate: HITLGate | None = None
    thread_repo: ThreadRepository | None = None
    parent_thread_id: UUID | None = None


class NotesAgent(StateflowAgent):
    """Personal-notes ``StateflowAgent``.

    Carries a ``NoteRepository`` so ``build_deps`` can mint a fresh
    ``NoteToolDeps`` per request, scoped to the requesting tenant.
    """

    name = "notes"
    metadata_model = None  # no per-thread settings yet

    def __init__(
        self,
        *,
        notes_repo: NoteRepository,
        hitl_gate: HITLGate | None = None,
        thread_repo: ThreadRepository | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._notes_repo = notes_repo
        self._hitl_gate = hitl_gate
        self._thread_repo = thread_repo
        self._model_name = model_name
        self._api_key = api_key

    def build_agent(self) -> Agent[NoteToolDeps, Any]:
        resolved_model = self._model_name or os.environ.get(
            "OPENROUTER_MODEL", DEFAULT_MODEL,
        )
        resolved_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY env var is required to build NotesAgent",
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
        return NoteToolDeps(
            repo=self._notes_repo,
            hitl_gate=self._hitl_gate,
            thread_repo=self._thread_repo,
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
) -> Note:
    """Open a confirmation thread, wait for user approval, then save the note.

    Use this INSTEAD of ``create_note`` when the user asks to create a
    TODO specifically (vs. a regular note). The flow:

    1. A NEW thread is opened, bound to the ``todo_approval`` agent.
    2. This run BLOCKS until the user confirms / rejects / modifies the
       proposal in the side thread.
    3. On approval the note is persisted with the (possibly modified)
       title / body. On rejection this tool raises and the note is
       NOT saved.

    Both ``title`` and ``body`` are required and free-form.
    """
    if (
        ctx.deps.hitl_gate is None
        or ctx.deps.thread_repo is None
        or ctx.deps.parent_thread_id is None
    ):
        raise RuntimeError(
            "propose_todo requires hitl_gate + thread_repo + parent_thread_id "
            "on NoteToolDeps — was NotesAgent constructed without them?",
        )

    hitl_gate = ctx.deps.hitl_gate
    thread_repo = ctx.deps.thread_repo

    prompt = HITLPrompt(
        title="Confirm todo",
        context=f"User wants to create a todo:\ntitle: {title!r}\nbody: {body!r}",
        decision_kinds={"approved", "rejected", "modified"},
    )

    # Spawn T2 first (so the frontend sees the side thread before the
    # gate workflow blocks on DBOS.recv). The approval agent in T2 reads
    # title/body out of thread metadata, NOT out of the user's first
    # message, so the model in T2 doesn't have to be told what's being
    # approved through chat history.
    request_id = uuid4()
    t2 = await thread_repo.create(
        agent="todo_approval",
        metadata={
            "request_id": str(request_id),
            "parent_thread_id": str(ctx.deps.parent_thread_id),
            "proposed_title": title,
            "proposed_body": body,
        },
    )
    # Seed an opening assistant message so the user sees something on
    # opening the side thread (otherwise it would look empty until they
    # type their first reply).
    await thread_repo.add_message(
        t2.id,
        role="assistant",
        parts=[{
            "type": "text",
            "text": f"Confirm todo: title={title!r}, body={body!r}?",
            "state": "done",
        }],
    )

    # Block until the approval thread fires DBOS.send. The UIChannel
    # simply awaits DBOS.recv("hitl:{request_id}") — whoever responds on
    # that topic unblocks the gate.
    response = await hitl_gate.run(prompt)

    if isinstance(response, ApprovedResponse):
        return await ctx.deps.repo.create(title=title, body=body)
    if isinstance(response, ModifiedResponse):
        mod = response.modified_proposal
        return await ctx.deps.repo.create(
            title=str(mod.get("title", title)),
            body=str(mod.get("body", body)),
        )
    # rejected
    raise RuntimeError("Todo creation rejected by user")


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

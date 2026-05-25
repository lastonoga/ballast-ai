"""OpenRouter-backed ``BallastAgent`` for the notes app.

One file = one agent. ``NotesAgent`` is the framework's per-thread agent
abstraction (see
``ballast.runtime.agents.BallastAgent``); the registry
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
from pydantic_ai.models.openrouter import OpenRouterModelSettings
from ballast.capabilities import (
    BallastCapability,
    BudgetGuard,
    JudgeAfterRun,
    LLMJudge,
    PIIGuard,
    RegexDetector,
)
from ballast.grounded import Ref, Selector
from ballast.persistence.thread.domain import Thread
from ballast.durable import Durable
from ballast.runtime import DurableAgent

from notes_app.agents.openrouter import (
    build_openrouter_model,
    default_model_settings,
)
from notes_app.agents.todo_approval import NotesTodoApprovalAgent
from notes_app.models.note import Note
from notes_app.models.todo_approval import TodoApprovalContext

DEFAULT_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are the assistant inside a personal notes app. "
    "You have tools to create, list, search, edit, and delete notes on "
    "the user's behalf, plus ``brainstorm_note`` for exploring ideas "
    "across multiple voices before saving. "
    "When the user asks you to create / find / change / remove a note, "
    "USE THE TOOLS to actually do it — do not just describe what you "
    "would do. Pick ``brainstorm_note`` when they want ideas, drafts, "
    "or exploration on a topic; pick ``create_note`` when they hand you "
    "a concrete note to save. After running the tools, briefly confirm "
    "what happened (e.g. 'Saved your note titled \"X\"'). "
    "If the user is chatting and not asking for a note action, just "
    "reply conversationally."
)


# Naive but useful PII patterns for the demo — apps with real privacy
# constraints would replace these with NER or a vetted policy library.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{8,}\d")


_QUALITY_JUDGE = LLMJudge(
    rubric=(
        "The assistant's reply addresses the user's actual request. "
        "Replies that ignore the user's intent, dodge the question, "
        "or reply with empty acknowledgements like 'sure' or 'ok' "
        "without doing anything DO NOT pass."
    ),
    threshold=0.6,
    max_retries=1,       # one retry on transient model error, then skip
)


def default_notes_capabilities() -> list[BallastCapability]:
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
    - ``JudgeAfterRun`` grades every assistant turn against a
      sanity-check rubric and pushes the verdict to the chat as a
      ``data-judge-verdict`` card (fire-and-forget; never blocks the
      user reply). ``fail_open=True`` (the default) means a judge
      model outage is logged + skipped, not user-visible.
    """
    caps: list[BallastCapability] = [
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
        JudgeAfterRun(
            _QUALITY_JUDGE,
            subject="assistant-turn",
            thread_id_from=lambda ctx: ctx.deps.parent_thread_id,
        ),
    ]
    return caps


@dataclass
class NoteToolDeps:
    """Per-request dependencies for the note tools.

    ``parent_thread_id`` is only used by ``propose_todo`` to spawn the
    durable approval workflow — the simpler note tools ignore it. May
    be ``None`` for tests that only exercise the non-HITL tools.

    Note repository, framework thread repo / event log / event stream
    — all reached via direct module-singleton imports (``notes_repo``)
    or ``ballast.get_ballast()``. Tests swap by monkeypatching the
    singleton module, NOT by passing alternative instances through
    these deps.
    """

    parent_thread_id: UUID | None = None


class NotesAgent(DurableAgent):
    """Personal-notes durable agent.

    Extends ``DurableAgent`` so the run loop is a
    ``@DBOS.workflow`` — survives SSE disconnects, process restarts,
    and resumable via Last-Event-ID. Tools / system_prompt / metadata
    semantics are identical to ``BallastAgent``; the only difference
    is the constructor (which now needs ``thread_repo`` + ``event_log``
    + ``event_stream`` for the durable infrastructure) and the run
    loop (which the streaming router dispatches into a workflow).
    """

    name = "notes"
    metadata_model = None  # no per-thread settings yet

    def build_agent(self) -> Agent[NoteToolDeps, Any]:
        # ``output_type=[str, DeferredToolRequests]`` opts into pydantic-ai's
        # deferred-tools branch: when the model calls a tool marked
        # ``requires_approval=True`` (e.g. ``delete_note``) the agent
        # pauses and yields a ``DeferredToolRequests`` instead of
        # looping forever over an unresolved tool call.
        return Agent(
            model=build_openrouter_model(),
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
        return NoteToolDeps(parent_thread_id=thread.id)

    def model_settings(self) -> OpenRouterModelSettings:
        """Hardcoded OpenRouter settings for the notes-app demo.

        The Alibaba-upstream ``content: null`` rejection (see
        ``KNOWN_BUGS.md`` B9) is fixed at the framework layer via
        ``AssistantMessageNormalizer`` — apps don't need to route
        around it here.
        """
        return default_model_settings(temperature=DEFAULT_TEMPERATURE)


# ── Tools ────────────────────────────────────────────────────────────────────
#
# Declared at module load via ``@NotesAgent.tool``. The framework
# registers them on the underlying pydantic-ai ``Agent`` the first time
# ``NotesAgent.agent`` is accessed, and auto-installs grounded
# ``prepare`` hooks for any ``Annotated[Ref[T], Selector(...)]`` params.


@NotesAgent.tool
async def create_note(
    ctx: RunContext[NoteToolDeps], title: str, body: str,
) -> str:
    """Save a note for the current user, gated by a UI approval card.

    Refines the draft, then suspends on ``create_note_flow``; the user
    sees the card in the side-panel "Approvals" drawer (separate from
    chat). On approve the note is persisted; on reject the save is
    cancelled. While waiting, a "Waiting for your approval →" pill
    appears in the chat thread.
    """
    from contextlib import nullcontext  # noqa: PLC0415

    from ballast.events import progress_to_thread  # noqa: PLC0415
    from ballast.patterns import ReflectionExhausted  # noqa: PLC0415

    from notes_app.agents.note_refiner import ProposedNote, note_refiner  # noqa: PLC0415
    from notes_app.workflows.create_note import create_note_flow  # noqa: PLC0415

    draft = ProposedNote(title=title, body=body)
    scope = (
        progress_to_thread(ctx.deps.parent_thread_id)
        if ctx.deps.parent_thread_id is not None
        else nullcontext()
    )
    with scope:
        if note_refiner is not None:
            try:
                refined = await note_refiner.run(draft)
            except ReflectionExhausted as exc:
                refined = exc.last_draft
        else:
            refined = draft

        note = await create_note_flow(refined)

    if note is None:
        return "Note save cancelled by user."
    return f"Saved note '{note.title}'."


@NotesAgent.tool
async def list_notes(
    ctx: RunContext[NoteToolDeps], limit: int = 20,
) -> list[Note]:
    """List the most recent notes for the current user, newest first.

    Use this when the user asks "show me my notes" or wants an
    overview. Returns at most ``limit`` notes (default 20).
    """
    del ctx
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415
    return await notes_repo.list_(limit=limit)


@NotesAgent.tool
async def search_notes(
    ctx: RunContext[NoteToolDeps], query: str, limit: int = 20,
) -> list[Note]:
    """Search the user's notes by case-insensitive substring on title+body.

    Returns matching notes newest-first, at most ``limit``. Use this
    when the user references a note by topic or keyword rather than id.
    """
    del ctx
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415
    return await notes_repo.search(query, limit=limit)


def _list_all_notes(_ctx: object) -> Any:
    """Selector source for ``Ref[Note]`` grounding — reads the module
    singleton so the Selector lambda doesn't need a deps shape."""
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415
    return notes_repo.list_(limit=1000)


@NotesAgent.tool
async def edit_note(
    ctx: RunContext[NoteToolDeps],
    note_id: Annotated[Ref[Note], Selector(_list_all_notes)],
    title: str | None = None,
    body: str | None = None,
) -> Note:
    """Edit an existing note. Pass only the fields you want to change.

    ``note_id`` is constrained at the schema level (via Selector) to
    the set of notes that currently exist for this user — you cannot
    fabricate one. Returns the updated note.
    """
    del ctx
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415
    nid = note_id.id if isinstance(note_id, Ref) else note_id
    return await notes_repo.update(nid, title=title, body=body)


@NotesAgent.tool
async def propose_todo(
    ctx: RunContext[NoteToolDeps], title: str, body: str,
) -> str:
    """Open a confirmation thread for a todo and return immediately.

    Tools run inline in the agent's DBOS workflow context (DurableAgent
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
    from uuid import uuid4  # noqa: PLC0415

    from dbos import SetWorkflowID  # noqa: PLC0415

    from ballast.events.context import progress_to_thread  # noqa: PLC0415
    from notes_app.workflows.todo_approval import todo_approval_flow  # noqa: PLC0415

    if ctx.deps.parent_thread_id is None:
        raise RuntimeError(
            "propose_todo requires parent_thread_id on NoteToolDeps "
            "— was NotesAgent.build_deps run without a thread?",
        )

    payload = TodoApprovalContext(
        proposed_title=title,
        proposed_body=body,
        parent_thread_id=ctx.deps.parent_thread_id,
    )
    with progress_to_thread(ctx.deps.parent_thread_id):
        with SetWorkflowID(str(uuid4())):
            await Durable.start_workflow(todo_approval_flow, payload)
    return (
        "I opened a confirmation thread to review this todo. "
        "Once you approve (or modify / reject) it there, I'll save "
        "it on this thread."
    )


@NotesAgent.tool
async def brainstorm_note(
    ctx: RunContext[NoteToolDeps], topic: str,
) -> str:
    """Brainstorm a note about a topic across multiple LLM voices, then
    ask the user which idea to keep.

    Fire-and-forget + durable: this tool returns IMMEDIATELY with a
    short acknowledgement. The brainstorm workflow runs in the
    background and emits typed progress events
    (``BrainstormChose`` / ``BrainstormSaved`` / ``BrainstormCancelled``
    / ``BrainstormTimedOut``) plus the underlying
    ``DivergentConvergent`` pattern's per-branch events — the chat
    renders all of those as cards live as they happen, including the
    approval prompt for the chosen idea.

    Use this when the user wants ideas or a draft on a topic
    (e.g. "brainstorm a todo about onboarding", "give me some ideas
    for tomorrow's standup"). For a single concrete note with no
    exploration, use ``create_note`` instead.

    Same ``(parent_thread, topic)`` collapses to one in-flight
    workflow — duplicate calls reattach instead of double-running.
    """
    from dbos import SetWorkflowID  # noqa: PLC0415

    from notes_app.models.brainstorm import BrainstormTask  # noqa: PLC0415
    from notes_app.workflows.brainstorm import (  # noqa: PLC0415
        brainstorm, workflow_id,
    )

    if ctx.deps.parent_thread_id is None:
        raise RuntimeError(
            "brainstorm_note requires parent_thread_id on NoteToolDeps "
            "— was NotesAgent.build_deps run without a thread?",
        )

    task = BrainstormTask(
        topic=topic, parent_thread_id=ctx.deps.parent_thread_id,
    )
    with SetWorkflowID(workflow_id(task)):
        await Durable.start_workflow(brainstorm, task)
    return (
        f"I started brainstorming ideas about {topic!r}. "
        "I'll show you the candidate ideas as they come in and ask "
        "which one to keep."
    )


@NotesAgent.tool(requires_approval=True)
async def delete_note(
    ctx: RunContext[NoteToolDeps],
    note_id: Annotated[Ref[Note], Selector(_list_all_notes)],
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
    del ctx
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415
    nid = note_id.id if isinstance(note_id, Ref) else note_id
    await notes_repo.delete(nid)
    return f"deleted {nid}"


# ── Module-level singleton ──────────────────────────────────────────────
# App-specific durable agent. Imported directly by ``main.py``'s
# dispatch table; per-request infra is bound on the instance via
# ``enqueue_run`` / ``enqueue_approval_resume`` from the supplied
# ``RunContext``.

notes_agent: NotesAgent = NotesAgent()

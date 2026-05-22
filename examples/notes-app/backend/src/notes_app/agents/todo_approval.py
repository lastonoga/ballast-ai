"""HITL approval ``BallastAgent`` for the notes-app todo-creation flow.

Threads bound to ``agent="todo_approval"`` are spawned by
``NotesAgent.propose_todo`` (see ``notes_app.agents.notes``). Each carries
metadata that tells THIS agent which durable approval workflow it's
gating + the proposed title/body from the parent thread.

The model running in this thread chats with the user and, when ready,
calls one of three tools — ``approve``, ``reject``, ``modify`` —
which forward an ``ApprovedResponse`` / ``RejectedResponse`` /
``ModifiedResponse`` to the durable approval workflow via
``DBOS.send``. The workflow (``TodoApprovalFlow.run`` in
``notes_app.workflows.todo_approval``) unblocks, saves the note (or
skips on reject), and posts a notification message back to the parent
thread.

This is the **durable** flavour of HITL — even if the parent thread's
SSE stream died before the user finished the approval here, the save
still happens because the parent run isn't in the loop anymore.

Note on annotations: like ``notes_app.agents.notes`` we do NOT use
``from __future__ import annotations`` so pydantic-ai's tool decoration
can resolve concrete types via ``get_type_hints()`` at module load.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ballast.durable import Durable
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from ballast.errors import MissingDependencyError
from ballast.patterns.hitl import (
    ApprovedResponse,
    ModifiedResponse,
    RejectedResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence.thread.domain import Thread
from ballast.runtime import BallastAgent

from notes_app.models.todo_approval import TodoApprovalContext
from notes_app.repositories.note import NoteRepository
from notes_app.settings import get_notes_settings

DEFAULT_MODEL = "qwen/qwen3.6-plus"
DEFAULT_TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are a confirmation helper. Discuss any concerns the user has, "
    "then call ONE of:\n"
    "  - approve()              — accept the todo as proposed\n"
    "  - reject(reason)         — cancel; reason is optional\n"
    "  - modify(new_title, new_body) — change title and/or body, then save\n"
    "Once you've called one of these, briefly confirm what happened.\n"
    "Be concise — this is a confirmation step, not a free chat."
)


@dataclass
class TodoApprovalDeps:
    """Per-request dependencies for the approve/reject/modify tools.

    ``workflow_id`` is the DBOS workflow id of the durable
    ``TodoApprovalFlow.run`` that's currently blocked on the HITL
    response. Helper tools send their decision there via
    ``Durable.send_async(destination_id=workflow_id, ...)``.
    """

    notes_repo: NoteRepository
    request_id: UUID
    workflow_id: str
    metadata: TodoApprovalContext


class NotesTodoApprovalAgent(BallastAgent):
    """Confirmation-helper ``BallastAgent`` for the notes-app todo flow.

    Constructor takes only the ``NoteRepository`` (currently unused by
    tools — they just route the decision to the durable workflow). The
    proposed title/body live on ``deps.metadata`` (typed
    ``TodoApprovalContext``); the framework injects them into the system
    prompt via the ``@system_prompt`` decorator below.
    """

    name = "todo_approval"
    metadata_model = TodoApprovalContext

    def build_agent(self) -> Agent[TodoApprovalDeps, Any]:
        settings = get_notes_settings()
        resolved_model = settings.openrouter_default_model or DEFAULT_MODEL
        resolved_key = (
            settings.openrouter_api_key.get_secret_value()
            if settings.openrouter_api_key else None
        )
        if not resolved_key:
            raise MissingDependencyError(
                "OpenRouter API key required to build NotesTodoApprovalAgent",
                hint="Set NOTES_APP_OPENROUTER_API_KEY (or legacy OPENROUTER_API_KEY) env var",
                context={"setting": "notes_app.openrouter_api_key"},
            )

        model = OpenRouterModel(
            resolved_model,
            provider=OpenRouterProvider(api_key=resolved_key),
        )
        return Agent(
            model=model,
            output_type=str,
            deps_type=TodoApprovalDeps,
            system_prompt=SYSTEM_PROMPT,
        )

    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> TodoApprovalDeps:
        del message
        # Direct import of the module-level singleton.
        from notes_app.repositories.note import notes_repo

        metadata = TodoApprovalContext.model_validate(thread.metadata_)
        return TodoApprovalDeps(
            notes_repo=notes_repo,
            request_id=UUID(thread.metadata_["request_id"]),
            workflow_id=str(thread.metadata_["workflow_id"]),
            metadata=metadata,
        )

    def model_settings(self) -> OpenRouterModelSettings:
        return OpenRouterModelSettings(
            temperature=DEFAULT_TEMPERATURE,
            openrouter_reasoning={"effort": "none"},
            openrouter_usage={"include": True},
        )


# ── System prompt: inject the typed context ─────────────────────────────────


@NotesTodoApprovalAgent.system_prompt
def _inject_todo_context(ctx: RunContext[TodoApprovalDeps]) -> str:
    return ctx.deps.metadata.to_system_prompt()


# ── Tools ────────────────────────────────────────────────────────────────────


async def _send_to_workflow(
    *,
    workflow_id: str,
    request_id: UUID,
    response: ApprovedResponse | RejectedResponse | ModifiedResponse,
) -> None:
    """Forward ``response`` to the durable ``TodoApprovalFlow.run`` workflow.

    Uses ``DBOS.send_async`` (the sync ``DBOS.send`` aborts with "called
    while an event loop is running" inside pydantic-ai's asyncio task
    on dbos 2.22+).
    """
    await Durable.send_async(
        destination_id=workflow_id,
        message=response.model_dump(mode="json"),
        topic=_hitl_topic(request_id),
    )


@NotesTodoApprovalAgent.tool
async def approve(ctx: RunContext[TodoApprovalDeps]) -> str:
    """Confirm the todo as proposed.

    Call this when the user agrees with the proposed title and body.
    No arguments — the proposal lives in thread metadata.
    """
    await _send_to_workflow(
        workflow_id=ctx.deps.workflow_id,
        request_id=ctx.deps.request_id,
        response=ApprovedResponse(
            actor_id="user", answered_at=datetime.now(tz=UTC),
        ),
    )
    return (
        f"Approved. Todo {ctx.deps.metadata.proposed_title!r} will be saved "
        "on the main thread."
    )


@NotesTodoApprovalAgent.tool
async def reject(ctx: RunContext[TodoApprovalDeps], reason: str = "") -> str:
    """Cancel the todo.

    Call this when the user changes their mind / says no. ``reason``
    is optional and carried through to the parent run for context.
    """
    await _send_to_workflow(
        workflow_id=ctx.deps.workflow_id,
        request_id=ctx.deps.request_id,
        response=RejectedResponse(
            actor_id="user",
            answered_at=datetime.now(tz=UTC),
            feedback=reason or "rejected",
        ),
    )
    return "Cancelled. The todo will NOT be saved."


@NotesTodoApprovalAgent.tool
async def modify(
    ctx: RunContext[TodoApprovalDeps],
    new_title: str | None = None,
    new_body: str | None = None,
) -> str:
    """Save the todo with a modified title and/or body.

    Missing fields fall back to the proposed defaults from the parent
    thread. Returns a short confirmation.
    """
    meta = ctx.deps.metadata
    final_title = new_title if new_title is not None else meta.proposed_title
    final_body = new_body if new_body is not None else meta.proposed_body
    await _send_to_workflow(
        workflow_id=ctx.deps.workflow_id,
        request_id=ctx.deps.request_id,
        response=ModifiedResponse(
            actor_id="user",
            answered_at=datetime.now(tz=UTC),
            feedback="",
            modified_proposal={"title": final_title, "body": final_body},
        ),
    )
    return (
        f"Updated to title={final_title!r}, body={final_body!r}. "
        "Saving on the main thread."
    )


# ── Module-level singleton ──────────────────────────────────────────────
# App-specific helper agent. Imported directly by ``main.py``'s
# dispatch table; the framework looks it up by ``Thread.agent ==
# "todo_approval"`` whenever the helper thread receives a message.

approval_agent: NotesTodoApprovalAgent = NotesTodoApprovalAgent()

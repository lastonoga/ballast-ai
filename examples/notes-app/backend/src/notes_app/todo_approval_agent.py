"""HITL approval ``StateflowAgent`` for the notes-app todo-creation flow.

Threads bound to ``agent="todo_approval"`` are spawned by
``NotesAgent.propose_todo`` (see ``notes_app/agent.py``). Each carries
metadata that tells THIS agent which durable approval workflow it's
gating + the proposed title/body from the parent thread.

The model running in this thread chats with the user and, when ready,
calls one of three tools — ``approve``, ``reject``, ``modify`` —
which forward an ``ApprovedResponse`` / ``RejectedResponse`` /
``ModifiedResponse`` to the durable approval workflow via
``DBOS.send``. The workflow (``TodoApprovalFlow.run`` in
``notes_app/todo_flow.py``) unblocks, saves the note (or skips on
reject), and posts a notification message back to the parent thread.

This is the **durable** flavour of HITL — even if the parent thread's
SSE stream died before the user finished the approval here, the save
still happens because the parent run isn't in the loop anymore.

Note on annotations: like ``agent.py`` we do NOT use
``from __future__ import annotations`` so pydantic-ai's tool decoration
can resolve concrete types via ``get_type_hints()`` at module load.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pydantic_ai_stateflow as sf
from dbos import DBOS

from pydantic_ai_stateflow.durable import Durable
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow.errors import MissingDependencyError
from pydantic_ai_stateflow.patterns.hitl import (
    ApprovedResponse,
    ModifiedResponse,
    RejectedResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.runtime import StateflowAgent

from notes_app.notes.repository import NoteRepository
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


class TodoApprovalContext(BaseModel):
    """Typed input context for the ``todo_approval`` agent.

    Plays a dual role:
      - ``StateflowAgent.metadata_model`` — validates ``Thread.metadata_``
        on thread creation (so the framework rejects malformed threads
        before they reach the agent).
      - Input contract for ``propose_todo`` → ``TodoApprovalFlow`` (the
        durable workflow gets a JSON-serialised instance of this class
        as its primary argument).

    Two framework-injected routing keys live on ``Thread.metadata_``
    alongside these fields (``request_id``, ``workflow_id``) — those
    are not modelled here because they're plumbing, not part of the
    user-facing context. The helper agent's ``build_deps`` reads them
    out of raw metadata.

    ``to_system_prompt`` projects the context into the agent's system
    prompt — SOLID: the context owns its own prompt projection.
    """

    proposed_title: str
    proposed_body: str
    parent_thread_id: UUID

    def to_system_prompt(self) -> str:
        return (
            "Review the proposed todo from the user's main notes thread:\n"
            f"  title: {self.proposed_title!r}\n"
            f"  body:  {self.proposed_body!r}"
        )

    def to_opening_message(self) -> str:
        """Initial assistant message seeded on the side thread."""
        return (
            f"Confirm todo: title={self.proposed_title!r}, "
            f"body={self.proposed_body!r}?"
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


@sf.stateflow_agent
class NotesTodoApprovalAgent(StateflowAgent):
    """Confirmation-helper ``StateflowAgent`` for the notes-app todo flow.

    Constructor takes only the ``NoteRepository`` (currently unused by
    tools — they just route the decision to the durable workflow). The
    proposed title/body live on ``deps.metadata`` (typed
    ``TodoApprovalContext``); the framework injects them into the system
    prompt via the ``@system_prompt`` decorator below.
    """

    name = "todo_approval"
    metadata_model = TodoApprovalContext

    def __init__(
        self,
        *,
        notes_repo: NoteRepository,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._notes_repo = notes_repo
        self._model_name = model_name
        self._api_key = api_key

    def build_agent(self) -> Agent[TodoApprovalDeps, Any]:
        settings = get_notes_settings()
        resolved_model = (
            self._model_name
            or settings.openrouter_default_model
            or DEFAULT_MODEL
        )
        resolved_key = (
            self._api_key
            or (settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else None)
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
        metadata = TodoApprovalContext.model_validate(thread.metadata_)
        return TodoApprovalDeps(
            notes_repo=self._notes_repo,
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

"""HITL approval ``BallastAgent`` for the notes-app todo-creation flow.

Threads bound to ``agent="todo_approval"`` are spawned by
``NotesAgent.propose_todo`` (see ``notes_app.agents.notes``). Each carries
metadata that tells THIS agent which durable approval workflow it's
gating + the proposed title/body from the parent thread.

The model running in this thread chats with the user and, when ready,
calls one of three tools — ``approve``, ``reject``, ``modify`` —
which build a ``CardVerdict[TodoApprovalContext]`` and forward it to the
durable ``todo_approval_flow`` workflow via ``Durable.send_async`` on the
``respond_topic`` stored in thread metadata. The workflow (in
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

from ballast.durable import Durable
from ballast.patterns.hitl.channels.ui_card import CardVerdict, register_card_kind
from ballast.persistence.thread.domain import Thread
from ballast.runtime import BallastAgent
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from notes_app.agents.openrouter import (
    build_openrouter_model,
    default_model_settings,
)
from notes_app.models.todo_approval import TodoApprovalContext

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

# Register the payload kind so the REST decision endpoint can validate
# incoming ``modified`` payloads against the right type.
register_card_kind(TodoApprovalContext)


@dataclass
class TodoApprovalDeps:
    """Per-request dependencies for the approve/reject/modify tools.

    ``workflow_id`` is the DBOS workflow id of the durable
    ``todo_approval_flow`` that's currently blocked on the HITL
    response. Helper tools send their decision there via
    ``Durable.send_async(destination_id=workflow_id, ...)``.

    ``respond_topic`` is the DBOS topic the workflow is listening on
    (stamped into thread metadata by ``ThreadChannel.deliver``).
    """

    workflow_id: str
    respond_topic: str
    metadata: TodoApprovalContext


class NotesTodoApprovalAgent(BallastAgent):
    """Confirmation-helper ``BallastAgent`` for the notes-app todo flow.

    The proposed title/body live on ``deps.metadata`` (typed
    ``TodoApprovalContext``); the framework injects them into the system
    prompt via the ``@system_prompt`` decorator below. Tools route
    decisions to the durable workflow via ``DBOS.send_async`` — no
    direct repo access needed (the workflow itself reaches the repo).
    """

    name = "todo_approval"
    metadata_model = TodoApprovalContext

    def build_agent(self) -> Agent[TodoApprovalDeps, Any]:
        return Agent(
            model=build_openrouter_model(),
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
        return TodoApprovalDeps(
            workflow_id=str(thread.metadata_["workflow_id"]),
            respond_topic=str(thread.metadata_["respond_topic"]),
            metadata=TodoApprovalContext.model_validate(thread.metadata_),
        )

    def model_settings(self) -> OpenRouterModelSettings:
        return default_model_settings(temperature=DEFAULT_TEMPERATURE)


# ── System prompt: inject the typed context ─────────────────────────────────


@NotesTodoApprovalAgent.system_prompt
def _inject_todo_context(ctx: RunContext[TodoApprovalDeps]) -> str:
    return ctx.deps.metadata.to_system_prompt()


# ── Tools ────────────────────────────────────────────────────────────────────


@NotesTodoApprovalAgent.tool
async def approve(ctx: RunContext[TodoApprovalDeps]) -> str:
    """Confirm the todo as proposed.

    Call this when the user agrees with the proposed title and body.
    No arguments — the proposal lives in thread metadata.
    """
    verdict: CardVerdict[TodoApprovalContext] = CardVerdict(
        decision="approve",
        modified=None,
        answered_at=datetime.now(tz=UTC),
    )
    await Durable.send_async(
        destination_id=ctx.deps.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=ctx.deps.respond_topic,
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
    verdict: CardVerdict[TodoApprovalContext] = CardVerdict(
        decision="reject",
        modified=None,
        feedback=reason or "rejected",
        answered_at=datetime.now(tz=UTC),
    )
    await Durable.send_async(
        destination_id=ctx.deps.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=ctx.deps.respond_topic,
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
    modified_payload = TodoApprovalContext(
        proposed_title=final_title,
        proposed_body=final_body,
        parent_thread_id=meta.parent_thread_id,
    )
    verdict: CardVerdict[TodoApprovalContext] = CardVerdict(
        decision="approve",
        modified=modified_payload,
        answered_at=datetime.now(tz=UTC),
    )
    await Durable.send_async(
        destination_id=ctx.deps.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=ctx.deps.respond_topic,
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

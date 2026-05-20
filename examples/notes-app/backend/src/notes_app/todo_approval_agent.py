"""HITL approval ``StateflowAgent`` for the notes-app todo-creation flow.

Threads bound to ``agent="todo_approval"`` are spawned by the
``propose_todo`` tool on ``NotesAgent`` (see ``notes_app/agent.py``).
The metadata they carry tells THIS agent which HITL request they're
gating and what title/body the user proposed in the parent thread.

The model running in this thread chats with the user and, when ready,
calls one of three tools — ``approve``, ``reject``, ``modify`` —
which forward an ``ApprovedResponse`` / ``RejectedResponse`` /
``ModifiedResponse`` to the gate's DBOS topic via ``DBOS.send``.
That unblocks the original ``NotesAgent`` run (which had called
``hitl_gate.run`` and was sitting on ``DBOS.recv``).

This is the **UIChannel** flavour of HITL — there is no
``DefaultHelperSessionRunner`` here. The framework's streaming router
handles the agent loop in T2 like any other thread; the side-thread is
just a normal chat with custom tools.

Note on annotations: like ``agent.py`` we do NOT use
``from __future__ import annotations`` so pydantic-ai's tool decoration
can resolve concrete types via ``get_type_hints()`` at module load.
"""

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from dbos import DBOS
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow.patterns.hitl import (
    ApprovedResponse,
    ModifiedResponse,
    RejectedResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence import HITLRepository
from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.runtime import StateflowAgent

from notes_app.notes.repository import NoteRepository

DEFAULT_MODEL = "qwen/qwen3.6-plus"
DEFAULT_TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are a confirmation helper. The user opened a side conversation "
    "to decide whether to commit a proposed todo from their main notes "
    "thread. Discuss any concerns they have, then call ONE of:\n"
    "  - approve()              — accept the todo as proposed\n"
    "  - reject(reason)         — cancel; reason is optional\n"
    "  - modify(new_title, new_body) — change title and/or body, then save\n"
    "Once you've called one of these, briefly confirm what happened.\n"
    "Be concise — this is a confirmation step, not a free chat."
)


class TodoApprovalMetadata(BaseModel):
    """Thread metadata schema for the ``todo_approval`` agent.

    Populated by ``NotesAgent.propose_todo`` when it spawns the side
    thread. Validated by the framework via
    ``StateflowAgent.metadata_model`` on thread creation.
    """

    request_id: UUID
    parent_thread_id: UUID
    proposed_title: str
    proposed_body: str


@dataclass
class TodoApprovalDeps:
    """Per-request dependencies for the approve/reject/modify tools."""

    notes_repo: NoteRepository
    hitl_repo: HITLRepository
    request_id: UUID
    proposed_title: str
    proposed_body: str


class NotesTodoApprovalAgent(StateflowAgent):
    """Confirmation-helper ``StateflowAgent`` for the notes-app todo flow.

    Constructor-injected with the same ``HITLRepository`` instance the
    notes-side ``HITLGate`` uses, plus a (currently-unused-by-tools)
    ``NoteRepository`` for symmetry with ``NotesAgent``. The model only
    needs the proposed title/body — pulled out of thread metadata in
    ``build_deps``.
    """

    name = "todo_approval"
    metadata_model = TodoApprovalMetadata

    def __init__(
        self,
        *,
        hitl_repo: HITLRepository,
        notes_repo: NoteRepository,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._hitl_repo = hitl_repo
        self._notes_repo = notes_repo
        self._model_name = model_name
        self._api_key = api_key

    def build_agent(self) -> Agent[TodoApprovalDeps, Any]:
        resolved_model = self._model_name or os.environ.get(
            "OPENROUTER_MODEL", DEFAULT_MODEL,
        )
        resolved_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY env var is required to build "
                "NotesTodoApprovalAgent",
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
        meta = TodoApprovalMetadata.model_validate(thread.metadata_)
        return TodoApprovalDeps(
            notes_repo=self._notes_repo,
            hitl_repo=self._hitl_repo,
            request_id=meta.request_id,
            proposed_title=meta.proposed_title,
            proposed_body=meta.proposed_body,
        )

    def model_settings(self) -> OpenRouterModelSettings:
        return OpenRouterModelSettings(
            temperature=DEFAULT_TEMPERATURE,
            openrouter_reasoning={"effort": "none"},
            openrouter_usage={"include": True},
        )


# ── Tools ────────────────────────────────────────────────────────────────────


async def _send_hitl_response(
    *,
    hitl_repo: HITLRepository,
    request_id: UUID,
    response: ApprovedResponse | RejectedResponse | ModifiedResponse,
) -> bool:
    """Look up the gate workflow id and forward ``response`` on the HITL topic.

    Returns ``True`` on success, ``False`` if the request is missing
    (e.g. expired / already responded) — callers turn that into a
    user-facing tool error string.

    Uses ``DBOS.send_async`` because the tool body runs inside the
    pydantic-ai agent's asyncio task — the sync ``DBOS.send`` aborts
    with "called while an event loop is running" in dbos 2.22+.
    """
    req = await hitl_repo.load_request(request_id)
    if req is None:
        return False
    await DBOS.send_async(
        destination_id=str(req.workflow_id),
        message=response.model_dump(mode="json"),
        topic=_hitl_topic(request_id),
    )
    return True


@NotesTodoApprovalAgent.tool
async def approve(ctx: RunContext[TodoApprovalDeps]) -> str:
    """Confirm the todo as proposed.

    Call this when the user agrees with the proposed title and body.
    No arguments — the proposal lives in thread metadata.
    """
    ok = await _send_hitl_response(
        hitl_repo=ctx.deps.hitl_repo,
        request_id=ctx.deps.request_id,
        response=ApprovedResponse(
            actor_id="user", answered_at=datetime.now(tz=UTC),
        ),
    )
    if not ok:
        return "Error: approval request not found (it may have expired)."
    return (
        f"Approved. Todo {ctx.deps.proposed_title!r} will be saved on the "
        "main thread."
    )


@NotesTodoApprovalAgent.tool
async def reject(ctx: RunContext[TodoApprovalDeps], reason: str = "") -> str:
    """Cancel the todo.

    Call this when the user changes their mind / says no. ``reason``
    is optional and carried through to the parent run for context.
    """
    ok = await _send_hitl_response(
        hitl_repo=ctx.deps.hitl_repo,
        request_id=ctx.deps.request_id,
        response=RejectedResponse(
            actor_id="user",
            answered_at=datetime.now(tz=UTC),
            feedback=reason or "rejected",
        ),
    )
    if not ok:
        return "Error: approval request not found (it may have expired)."
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
    final_title = new_title if new_title is not None else ctx.deps.proposed_title
    final_body = new_body if new_body is not None else ctx.deps.proposed_body
    ok = await _send_hitl_response(
        hitl_repo=ctx.deps.hitl_repo,
        request_id=ctx.deps.request_id,
        response=ModifiedResponse(
            actor_id="user",
            answered_at=datetime.now(tz=UTC),
            feedback="",
            modified_proposal={"title": final_title, "body": final_body},
        ),
    )
    if not ok:
        return "Error: approval request not found (it may have expired)."
    return (
        f"Updated to title={final_title!r}, body={final_body!r}. "
        "Saving on the main thread."
    )

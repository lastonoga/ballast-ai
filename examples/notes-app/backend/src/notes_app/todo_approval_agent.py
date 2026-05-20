"""HITL approval ``StateflowAgent`` for the notes-app todo-creation flow.

Threads bound to ``agent="todo_approval"`` are spawned by
``HITLGate.ask_helper`` when ``NotesAgent.propose_todo`` asks for a
decision. The metadata they carry tells THIS agent which HITL request
they're gating and what title/body the user proposed in the parent
thread.

The model running in this thread chats with the user and, when ready,
calls one of three tools — ``approve``, ``reject``, ``modify`` —
which forward an ``ApprovedResponse`` / ``RejectedResponse`` /
``ModifiedResponse`` to the gate's DBOS topic via ``DBOS.send``.
That unblocks the original ``NotesAgent`` run (which had called
``hitl_gate.ask_helper`` and was sitting on ``DBOS.recv``).

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
      - Input contract for ``HITLGate.ask_helper(context=...)`` — the
        caller builds an instance and the gate puts it on the helper
        thread's metadata.

    The ``request_id`` (HITL routing key) is NOT a field here — the
    framework injects it into thread metadata alongside this model's
    fields when the helper thread is created. The helper agent's
    ``build_deps`` reads it back as a separate value.

    ``to_system_prompt`` projects the context into the agent's system
    prompt — SOLID: the context owns its own prompt projection, so the
    agent class doesn't need a separate template. The
    ``@NotesTodoApprovalAgent.system_prompt`` decorator below wires it
    up at module load.
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
        """Initial assistant message seeded on the side thread.

        Shown to the user the moment they open the side thread, before
        the helper agent's first model call. Kept short and structural;
        the model writes its own follow-up on the first user turn.
        """
        return (
            f"Confirm todo: title={self.proposed_title!r}, "
            f"body={self.proposed_body!r}?"
        )


@dataclass
class TodoApprovalDeps:
    """Per-request dependencies for the approve/reject/modify tools."""

    notes_repo: NoteRepository
    hitl_repo: HITLRepository
    request_id: UUID
    metadata: TodoApprovalContext


class NotesTodoApprovalAgent(StateflowAgent):
    """Confirmation-helper ``StateflowAgent`` for the notes-app todo flow.

    Constructor-injected with the same ``HITLRepository`` instance the
    notes-side ``HITLGate`` uses, plus a (currently-unused-by-tools)
    ``NoteRepository`` for symmetry with ``NotesAgent``. The proposed
    title/body live on ``deps.metadata`` (typed
    ``TodoApprovalContext``); the framework injects them into the system
    prompt via the ``@system_prompt`` decorator below.
    """

    name = "todo_approval"
    metadata_model = TodoApprovalContext

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
        # ``request_id`` is a framework-injected routing key sitting next
        # to the user-shaped context fields. ``metadata_model`` validates
        # the dict with ``extra="ignore"`` (pydantic v2 default), so the
        # extra key doesn't trip validation.
        metadata = TodoApprovalContext.model_validate(thread.metadata_)
        request_id = UUID(thread.metadata_["request_id"])
        return TodoApprovalDeps(
            notes_repo=self._notes_repo,
            hitl_repo=self._hitl_repo,
            request_id=request_id,
            metadata=metadata,
        )

    def model_settings(self) -> OpenRouterModelSettings:
        return OpenRouterModelSettings(
            temperature=DEFAULT_TEMPERATURE,
            openrouter_reasoning={"effort": "none"},
            openrouter_usage={"include": True},
        )


# ── System prompt: inject the typed context ─────────────────────────────────
#
# The base ``SYSTEM_PROMPT`` above is generic ("you are a confirmation
# helper, call approve/reject/modify"); the per-thread context (which
# todo is being reviewed) lives on ``deps.metadata`` and is appended
# here. Following the SOLID brief, the projection logic lives ON the
# context model (``TodoApprovalContext.to_system_prompt``) — this
# decorator just plumbs it through.


@NotesTodoApprovalAgent.system_prompt
def _inject_todo_context(ctx: RunContext[TodoApprovalDeps]) -> str:
    return ctx.deps.metadata.to_system_prompt()


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
        f"Approved. Todo {ctx.deps.metadata.proposed_title!r} will be saved on "
        "the main thread."
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
    meta = ctx.deps.metadata
    final_title = new_title if new_title is not None else meta.proposed_title
    final_body = new_body if new_body is not None else meta.proposed_body
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

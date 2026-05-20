"""Durable todo-approval workflow.

Fire-and-forget from ``NotesAgent.propose_todo``: the tool creates the
helper thread + an empty opening message, kicks off this workflow via
``DBOS.start_workflow_async``, and returns immediately with a friendly
"I opened a side conversation" string. The model in T1 streams that
out and the request handler can die — the workflow survives in DBOS.

When the helper agent in T2 calls ``approve`` / ``modify`` / ``reject``,
its tools ``DBOS.send`` the response to THIS workflow's id (stored on
T2's metadata). The workflow's blocking ``DBOS.recv_async`` unblocks,
the note is saved (or skipped on rejection), and a notification message
is appended to the parent thread — so when the user comes back to T1
(or never left), they see "Saved your todo titled …" without needing
the original SSE stream to have stayed alive.

Note on annotations: like ``agent.py`` we do NOT use
``from __future__ import annotations`` so DBOS / pydantic-ai can resolve
concrete types at decoration time.
"""

import itertools
from typing import Any
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance
from pydantic import TypeAdapter

from pydantic_ai_stateflow.patterns.hitl.response import (
    ApprovedResponse,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

from notes_app.notes.repository import NoteRepository
from notes_app.todo_approval_agent import TodoApprovalContext

_counter = itertools.count()
_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)

# Effectively "wait forever" without violating ``DBOS.recv_async``'s
# arithmetic. Apps that need an actual deadline pass it explicitly.
_NO_TIMEOUT_SECONDS: float = 365 * 24 * 60 * 60.0


@DBOS.dbos_class()
class TodoApprovalFlow(DBOSConfiguredInstance):
    """Durable container for the todo-approval workflow.

    ``DBOSConfiguredInstance`` lets DBOS rehydrate the instance during
    workflow recovery by remembering its ``config_name`` → repos
    binding. Apps construct one instance at boot and register it with
    DBOS via the framework's ``Engine`` lifecycle (or directly inside a
    ``DBOS.launch()`` block).
    """

    def __init__(
        self,
        *,
        notes_repo: NoteRepository,
        thread_repo: ThreadRepository,
    ) -> None:
        super().__init__(config_name=f"todo-approval-flow-{next(_counter)}")
        self.notes_repo = notes_repo
        self.thread_repo = thread_repo

    @DBOS.workflow()
    async def run(
        self,
        *,
        context_dict: dict[str, Any],
        request_id: str,
    ) -> None:
        """Block on the helper's HITL response, then save + notify.

        ``context_dict`` is the JSON-serialized ``TodoApprovalContext``
        (DBOS pickles workflow args by default but JSON-shaped dicts are
        safest across serializer changes).
        """
        context = TodoApprovalContext.model_validate(context_dict)
        rid = UUID(request_id)
        topic = _hitl_topic(rid)

        payload = await DBOS.recv_async(
            topic, timeout_seconds=_NO_TIMEOUT_SECONDS,
        )
        if payload is None:
            await self._notify_parent(
                context.parent_thread_id,
                "Todo approval timed out — nothing was saved.",
            )
            return

        response = _RESPONSE_ADAPTER.validate_python(payload)

        if isinstance(response, ApprovedResponse):
            note = await self.notes_repo.create(
                title=context.proposed_title,
                body=context.proposed_body,
            )
            await self._notify_parent(
                context.parent_thread_id,
                f"Saved your todo titled {note.title!r}.",
            )
        elif isinstance(response, ModifiedResponse):
            mod = response.modified_proposal
            title = str(mod.get("title", context.proposed_title))
            body = str(mod.get("body", context.proposed_body))
            note = await self.notes_repo.create(title=title, body=body)
            await self._notify_parent(
                context.parent_thread_id,
                f"Saved your todo titled {note.title!r} (with your edits).",
            )
        elif isinstance(response, RejectedResponse):
            reason = (response.feedback or "").strip()
            tail = f" ({reason})" if reason else ""
            await self._notify_parent(
                context.parent_thread_id,
                f"Todo creation was cancelled{tail}.",
            )
        else:
            # TimeoutResponse / unknown — defensive fallthrough; the
            # recv None-branch above already handles real timeouts.
            await self._notify_parent(
                context.parent_thread_id,
                "Todo approval ended without a decision.",
            )

    async def _notify_parent(self, parent_id: UUID, text: str) -> None:
        await self.thread_repo.add_message(
            parent_id,
            role="assistant",
            parts=[{"type": "text", "text": text, "state": "done"}],
        )

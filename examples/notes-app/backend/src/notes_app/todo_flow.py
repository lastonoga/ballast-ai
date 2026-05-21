"""Notes-app's concrete durable HITL workflow.

Subclasses ``DurableHITLWorkflow`` from the framework — the framework
owns thread spawn, workflow lifecycle, ``DBOS.recv_async`` blocking,
and context rehydration. This module just supplies the
``on_decision`` body: save the note (or skip on reject) and post a
notification message back to the parent thread.

Note on annotations: like ``agent.py`` we do NOT use
``from __future__ import annotations`` so DBOS / pydantic-ai can
resolve concrete types at decoration time.
"""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel
from pydantic_ai_stateflow.patterns.hitl import (
    ApprovedResponse,
    DurableHITLWorkflow,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
)
from pydantic_ai_stateflow.persistence.events.repository import (
    EventLogRepository,
)
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
from pydantic_ai_stateflow.runtime.event_stream import (
    EventNotification,
    EventStream,
    thread_channel,
)

from notes_app.notes.repository import NoteRepository
from notes_app.todo_approval_agent import TodoApprovalContext


class TodoApprovalFlow(DurableHITLWorkflow):
    """Save-on-approve / notify-on-reject post-decision logic for todos.

    Fully durable: the workflow body runs inside DBOS so the note save
    and the parent-thread notification both happen even if T1's
    streaming request handler died long before the helper agent
    finished its conversation.

    ``event_log`` / ``event_stream`` (optional) — when wired, ``_notify``
    additionally emits a ``message-added`` event into the parent
    thread's event log + publishes through the signal channel. Any
    frontend tailing ``GET /threads/{parent_id}/events`` picks the
    new message up live without a page reload.
    """

    def __init__(
        self,
        *,
        notes_repo: NoteRepository,
        thread_repo: ThreadRepository,
        event_log: Optional[EventLogRepository] = None,
        event_stream: Optional[EventStream] = None,
        config_name: str = "notes-todo-approval-flow",
    ) -> None:
        # Stable ``config_name`` so DBOS can rebind this instance to its
        # in-flight workflows after a restart — apps construct ONE
        # TodoApprovalFlow at boot and re-construct it with the same
        # name on recovery (otherwise DBOS can't address the instance).
        # Tests override the default to keep per-test instances unique.
        super().__init__(
            thread_repo=thread_repo, config_name=config_name,
        )
        self.notes_repo = notes_repo
        self._event_log = event_log
        self._event_stream = event_stream

    async def on_decision(
        self,
        *,
        response: HITLResponse,
        context: BaseModel,
    ) -> None:
        assert isinstance(context, TodoApprovalContext), (
            f"Expected TodoApprovalContext, got {type(context).__name__}"
        )
        parent_id = context.parent_thread_id

        if isinstance(response, ApprovedResponse):
            note = await self.notes_repo.create(
                title=context.proposed_title,
                body=context.proposed_body,
            )
            await self._notify(
                parent_id, f"Saved your todo titled {note.title!r}.",
            )
        elif isinstance(response, ModifiedResponse):
            mod = response.modified_proposal
            title = str(mod.get("title", context.proposed_title))
            body = str(mod.get("body", context.proposed_body))
            note = await self.notes_repo.create(title=title, body=body)
            await self._notify(
                parent_id,
                f"Saved your todo titled {note.title!r} (with your edits).",
            )
        elif isinstance(response, RejectedResponse):
            reason = (response.feedback or "").strip()
            tail = f" ({reason})" if reason else ""
            await self._notify(
                parent_id, f"Todo creation was cancelled{tail}.",
            )
        else:
            # TimeoutResponse / unknown
            await self._notify(
                parent_id, "Todo approval timed out — nothing was saved.",
            )

    async def _notify(self, parent_id: UUID, text: str) -> None:
        msg = await self.thread_repo.add_message(
            parent_id,
            role="assistant",
            parts=[{"type": "text", "text": text, "state": "done"}],
        )
        # Push a ``message-added`` event into the parent thread's event
        # log + signal channel so any open ``GET /threads/{id}/events``
        # SSE consumer (frontend tailing for cross-workflow notifications)
        # receives it live without a page reload. No-op when wiring
        # isn't provided (e.g. in unit tests).
        if self._event_log is None:
            return
        ev = await self._event_log.append(
            thread_id=parent_id,
            kind="message-added",
            payload={
                "id": msg.id,
                "role": msg.role,
                "parts": msg.parts,
            },
        )
        if self._event_stream is not None:
            await self._event_stream.publish(
                thread_channel(parent_id),
                EventNotification(thread_id=parent_id, seq=ev.seq),
            )

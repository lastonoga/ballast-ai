"""Notes-app's concrete durable HITL workflow.

Subclasses ``DurableHITLWorkflow`` from the framework — the framework
owns thread spawn, workflow lifecycle, ``DBOS.recv_async`` blocking,
and context rehydration. This module just supplies the
``on_decision`` body: save the note (or skip on reject) and post a
notification message back to the parent thread.

Infra (``thread_repo`` / ``event_log`` / ``event_stream``) is supplied
per-call via the ``RunContext`` passed to ``open(ctx, ...)`` — the
base class binds it on the instance so the durable workflow body
(and ``_notify`` below) can reach the right repos for this call.

Note on annotations: like ``notes_app.agents.notes`` we do NOT use
``from __future__ import annotations`` so DBOS / pydantic-ai can
resolve concrete types at decoration time.
"""

from uuid import UUID

from pydantic import BaseModel
from pydantic_ai_stateflow.patterns.hitl import (
    ApprovedResponse,
    DurableHITLWorkflow,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
)
from pydantic_ai_stateflow.runtime.event_stream import (
    EventNotification,
    thread_channel,
)

from notes_app.models.todo_approval import TodoApprovalContext


class TodoApprovalFlow(DurableHITLWorkflow):
    """Save-on-approve / notify-on-reject post-decision logic for todos.

    Fully durable: the workflow body runs inside DBOS so the note save
    and the parent-thread notification both happen even if T1's
    streaming request handler died long before the helper agent
    finished its conversation.

    Per-call infra (thread repo + event log + event stream) is bound
    on the instance via ``open(ctx, ...)`` — the base class assigns
    ``self.thread_repo`` / ``self._event_log`` / ``self._event_stream``
    from ``ctx`` before the workflow body executes. ``_notify``
    consults those slots to persist the parent-thread notification
    and broadcast a ``message-added`` event.
    """

    async def on_decision(
        self,
        *,
        response: HITLResponse,
        context: BaseModel,
    ) -> None:
        # Lazy import of the module-level singleton — tests
        # monkeypatch ``notes_app.repositories.note.notes_repo`` so the
        # swap is visible here on the per-test fixture.
        from notes_app.repositories.note import notes_repo

        assert isinstance(context, TodoApprovalContext), (
            f"Expected TodoApprovalContext, got {type(context).__name__}"
        )
        parent_id = context.parent_thread_id

        if isinstance(response, ApprovedResponse):
            note = await notes_repo.create(
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
            note = await notes_repo.create(title=title, body=body)
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


# ── Module-level singleton ──────────────────────────────────────────────
# App-specific durable HITL workflow. Imported directly by callers that
# need to spawn approval flows (NotesAgent.propose_todo, BrainstormFlow).

todo_flow: TodoApprovalFlow = TodoApprovalFlow(
    config_name="notes-todo-approval-flow",
)

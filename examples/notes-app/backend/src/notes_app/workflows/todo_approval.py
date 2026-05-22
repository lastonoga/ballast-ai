"""Notes-app's concrete durable HITL workflow.

Subclasses ``DurableHITLWorkflow`` from the framework — the framework
owns thread spawn, workflow lifecycle, ``DBOS.recv_async`` blocking,
and context rehydration. This module just supplies the
``on_decision`` body: save the note (or skip on reject) and post a
notification message back to the parent thread.

Repos + stream are reached via the process-wide ``Engine`` installed
by ``ballast.create_app`` at startup — no per-call ``RunContext`` is
threaded through ``open(...)`` anymore.

Note on annotations: like ``notes_app.agents.notes`` we do NOT use
``from __future__ import annotations`` so DBOS / pydantic-ai can
resolve concrete types at decoration time.
"""

from uuid import UUID

from pydantic import BaseModel
from ballast import get_ballast
from ballast.patterns.hitl import (
    ApprovedResponse,
    DurableHITLWorkflow,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
)
from ballast.runtime.event_stream import (
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

    Reaches the thread repo + event log + event stream through
    ``ballast.get_ballast()`` — the framework owns the singleton wired by
    ``ballast.create_app`` at startup, so ``_notify`` doesn't need per-call
    infra injection.
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
        engine = get_ballast()
        msg = await engine.thread_repo.add_message(
            parent_id,
            role="assistant",
            parts=[{"type": "text", "text": text, "state": "done"}],
        )
        # Push a ``message-added`` event into the parent thread's event
        # log + signal channel so any open ``GET /threads/{id}/events``
        # SSE consumer (frontend tailing for cross-workflow notifications)
        # receives it live without a page reload.
        ev = await engine.event_log.append(
            thread_id=parent_id,
            kind="message-added",
            payload={
                "id": msg.id,
                "role": msg.role,
                "parts": msg.parts,
            },
        )
        await engine.event_stream.publish(
            thread_channel(parent_id),
            EventNotification(thread_id=parent_id, seq=ev.seq),
        )


# ── Module-level singleton ──────────────────────────────────────────────
# App-specific durable HITL workflow. Imported directly by callers that
# need to spawn approval flows (NotesAgent.propose_todo, BrainstormFlow).

todo_flow: TodoApprovalFlow = TodoApprovalFlow()

"""Notes-app's durable todo-approval workflow.

Uses ``ThreadChannel`` to open a helper sub-thread with the
``NotesTodoApprovalAgent``, then dispatches on the typed
``CardVerdict[TodoApprovalContext]`` that comes back.

Note on annotations: like ``notes_app.agents.notes`` we do NOT use
``from __future__ import annotations`` so DBOS / pydantic-ai can
resolve concrete types at decoration time.
"""

from uuid import UUID

from ballast.durable import Durable
from ballast.patterns.hitl.channels.thread import ThreadChannel
from ballast.patterns.hitl.channels.ui_card import CardVerdict

from notes_app.agents.todo_approval import NotesTodoApprovalAgent
from notes_app.models.todo_approval import TodoApprovalContext

# ── Channel ────────────────────────────────────────────────────────────────
# Must be at module level so it is wired before DBOS.launch().

_channel: ThreadChannel[TodoApprovalContext] = ThreadChannel(
    helper_agent=NotesTodoApprovalAgent,
    payload_type=TodoApprovalContext,
    opening_message=(
        "I'm proposing a note. Use approve / reject / modify to decide."
    ),
)


# ── Workflow ───────────────────────────────────────────────────────────────


@Durable.workflow()
async def todo_approval_flow(payload: TodoApprovalContext) -> None:
    """Durable workflow: block on a helper thread's verdict, then act.

    Fire-and-forget from the parent agent's perspective. The workflow
    survives process restarts because DBOS persists every step.
    """
    try:
        verdict: CardVerdict[TodoApprovalContext] = await _channel.request(payload)
    except TimeoutError:
        await _notify(
            payload.parent_thread_id,
            "Todo approval timed out — nothing was saved.",
        )
        return

    if verdict.decision == "approve":
        final = verdict.modified or payload
        # Lazy import — tests monkeypatch the module singleton.
        from notes_app.repositories.note import notes_repo  # noqa: PLC0415

        note = await notes_repo.create(
            title=final.proposed_title,
            body=final.proposed_body,
        )
        edits = " (with your edits)" if verdict.modified is not None else ""
        await _notify(
            payload.parent_thread_id,
            f"Saved your todo titled {note.title!r}{edits}.",
        )
    else:  # reject
        reason = (verdict.feedback or "").strip()
        tail = f" ({reason})" if reason else ""
        await _notify(
            payload.parent_thread_id,
            f"Todo creation was cancelled{tail}.",
        )


async def _notify(parent_id: UUID, text: str) -> None:
    """Post a notification message back to the parent thread."""
    from ballast import get_ballast  # noqa: PLC0415

    engine = get_ballast()
    await engine.thread_repo.add_message(
        parent_id,
        role="assistant",
        parts=[{"type": "text", "text": text, "state": "done"}],
    )

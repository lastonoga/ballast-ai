"""The ``brainstorm`` durable workflow — diverge → ask user → save.

Reads top-to-bottom as the imperative pipeline it is:

  1. ``progress_to_thread`` scope routes pattern + workflow progress
     into the parent chat as typed UI events.
  2. ``divergent.run(topic)`` fans out to N LLM agents and picks one
     ``TodoIdea``.
  3. ``_brainstorm_channel.request(...)`` opens a helper thread and
     blocks for the user's verdict (CardVerdict).
  4. Verdict handling persists the note (or skips) and emits a
     workflow-level event (``BrainstormSaved`` / ``BrainstormCancelled``
     / ``BrainstormTimedOut``) that the default chat router renders.

All side effects (note save, signal emits, the chat-handler writes)
sit behind DBOS step boundaries so a crash mid-flow recovers cleanly
on restart.

No ``from __future__ import annotations``: pydantic-ai introspects
``get_type_hints()`` at decoration time (same as
``notes_app.agents.notes``).
"""

from ballast import Durable
from ballast.events import progress_to_thread
from ballast.patterns.hitl.channels import CardVerdict, ThreadChannel

from notes_app.agents.todo_approval import NotesTodoApprovalAgent
from notes_app.models.brainstorm import BrainstormOutcome, BrainstormTask
from notes_app.models.todo import TodoIdea
from notes_app.models.todo_approval import TodoApprovalContext
from notes_app.workflows.brainstorm.divergent import divergent
from notes_app.workflows.brainstorm.events import (
    BrainstormCancelled,
    BrainstormChose,
    BrainstormSaved,
    BrainstormTimedOut,
    brainstorm_progress,
)

# Module-level singleton channel — no static opening_message because
# the opening text is dynamic (contains the proposed title/body).
# The helper agent's system prompt already receives the full context
# via TodoApprovalContext.to_system_prompt(), so no UX is lost.
_brainstorm_channel: ThreadChannel[TodoApprovalContext] = ThreadChannel(
    helper_agent=NotesTodoApprovalAgent,
    payload_type=TodoApprovalContext,
)


def workflow_id(task: BrainstormTask) -> str:
    """Deterministic workflow id for the HTTP route.

    Same ``(parent_thread, topic)`` → same workflow id so duplicate
    clicks collapse to one in-flight workflow."""
    return f"brainstorm:{task.parent_thread_id}:{abs(hash(task.topic))}"


@Durable.workflow()
async def brainstorm(task: BrainstormTask) -> BrainstormOutcome:
    """Diverge → converge → ask user → save (or not).

    See module docstring for the step-by-step shape. Each phase
    publishes a typed signal event; the default chat router (auto-
    connected in ``events.py``) turns those into ``data-<event>``
    chat parts the frontend renders with bespoke components.
    """
    parent_thread_id = task.parent_thread_id
    topic = task.topic

    with progress_to_thread(parent_thread_id):
        chosen: TodoIdea = await divergent.run(topic)

        await brainstorm_progress.send(
            sender=None, event=BrainstormChose(title=chosen.title),
        )

        approval_context = TodoApprovalContext(
            proposed_title=chosen.title,
            proposed_body=chosen.body,
            parent_thread_id=parent_thread_id,
        )

        try:
            verdict: CardVerdict[TodoApprovalContext] = await _brainstorm_channel.request(
                approval_context,
            )
        except TimeoutError:
            await brainstorm_progress.send(
                sender=None, event=BrainstormTimedOut(),
            )
            return BrainstormOutcome(
                proposed_title=chosen.title,
                proposed_body=chosen.body,
                saved_title=None,
                saved_body=None,
            )

        saved_title: str | None = None
        saved_body: str | None = None

        if verdict.decision == "approve":
            final = verdict.modified or approval_context
            note = await _save_note(title=final.proposed_title, body=final.proposed_body)
            saved_title, saved_body = note.title, note.body
            await brainstorm_progress.send(
                sender=None,
                event=BrainstormSaved(
                    title=note.title,
                    modified=verdict.modified is not None,
                ),
            )
        else:  # reject
            await brainstorm_progress.send(
                sender=None,
                event=BrainstormCancelled(reason=verdict.feedback or None),
            )

    return BrainstormOutcome(
        proposed_title=chosen.title,
        proposed_body=chosen.body,
        saved_title=saved_title,
        saved_body=saved_body,
    )


@Durable.step()
async def _save_note(*, title: str, body: str):  # noqa: ANN201 — domain type
    """Persist a note via the module-level singleton repo.

    ``@Durable.step``-wrapped so workflow replay sees the same note id
    (DBOS memoises step return values by step name + args) instead of
    double-creating on crash recovery.
    """
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415
    return await notes_repo.create(title=title, body=body)


__all__ = ["brainstorm", "workflow_id"]

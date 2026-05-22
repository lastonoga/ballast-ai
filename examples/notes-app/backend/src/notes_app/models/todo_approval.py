"""Typed context model for the todo-approval helper agent."""

from uuid import UUID

from pydantic import BaseModel


class TodoApprovalContext(BaseModel):
    """Typed input context for the ``todo_approval`` agent.

    Plays a dual role:
      - ``BallastAgent.metadata_model`` — validates ``Thread.metadata_``
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


__all__ = ["TodoApprovalContext"]

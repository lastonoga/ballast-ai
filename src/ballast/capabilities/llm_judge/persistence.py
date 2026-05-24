"""Opt-in helper that writes a judge verdict into a thread's event log.

Kept separate from :class:`LLMJudge` because judges grade everything
(plans, retrievals, tool args, agent turns) but only some grading
sites want the verdict surfaced in the user-facing chat. Routing
through :class:`ThreadEventBroadcaster` reuses the same payload
contract that `message-added` events do, so the wire shape stays
leak-proof end-to-end.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from ballast.capabilities.llm_judge._models import (
        JudgeVerdict,
        PairwiseVerdict,
    )


async def persist_verdict_as_thread_event(
    thread_id: UUID,
    verdict: "JudgeVerdict | PairwiseVerdict",
    *,
    subject: str,
) -> None:
    """Write a judge verdict to the thread's event log as a
    ``data-judge-verdict`` UI card.

    Callers decide WHEN to persist — every site that wants the
    verdict surfaced calls this explicitly. Not auto-wired so judges
    used for fire-and-forget telemetry stay free of I/O.

    ``subject`` is a free-form label (``"assistant-turn"``,
    ``"tool-call:create_note"``, ``"retrieved-chunk:42"``) that
    explains WHAT was graded — the verdict on its own is meaningless
    without the subject.
    """
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    engine = get_ballast()
    # Broadcaster routes through ``MessageAddedPayload`` so the wire
    # shape can't drift — same contract as any other ``data-*`` UI card.
    await engine.broadcaster.emit_raw(
        thread_id,
        part={
            "type": "data-judge-verdict",
            "data": {
                "subject": subject,
                **verdict.model_dump(mode="json", by_alias=True),
            },
        },
        persistent=True,
    )


__all__ = ["persist_verdict_as_thread_event"]

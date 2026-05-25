"""``create_note_flow`` — child ``@Durable.workflow`` that gates note
persistence behind a UI-card approval.

Extracted from the tool body so the human-pause unit is its own DBOS
workflow (visible independently in the inspector; recoverable on
restart without re-driving the entire agent run).
"""
from __future__ import annotations

import logging

from ballast import get_ballast
from ballast.durable import Durable
from ballast.memory.episodic import DetailLevel
from ballast.memory.episodic.strategies import TopK
from ballast.patterns.hitl.channels.ui_card import (
    UICardChannel, register_card_kind,
)

from notes_app.agents.note_refiner import ProposedNote
from notes_app.models.note import Note

register_card_kind(ProposedNote)

_log = logging.getLogger(__name__)

# Channel needs the payload type to decode the verdict's `modified`
# field into a typed ProposedNote on resume. Constructed once at module
# import; safe to share across invocations (stateless besides the
# captured payload type).
_channel: UICardChannel[ProposedNote] = UICardChannel(payload_type=ProposedNote)


@Durable.workflow()
async def create_note_flow(draft: ProposedNote) -> Note | None:
    """Ask the user to approve persisting ``draft``; save on approve,
    return ``None`` on reject."""
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415

    # Best-effort recall: similar past notes for context. Failures don't
    # block the save flow.
    try:
        memory = getattr(get_ballast(), "_episodic_memory", None)
        if memory is not None:
            recall = await memory.episodic_for(
                intent=f"prior notes about {draft.title}",
                strategy=TopK(k=3, detail=DetailLevel.PREVIEW),
            )
            _log.info(
                "create_note_flow: recall returned %d similar (top: %r)",
                len(recall.episodes),
                recall.episodes[0].episode.preview if recall.episodes else None,
            )
    except Exception:
        _log.exception("memory recall failed (continuing without context)")

    verdict = await _channel.request(draft)
    if verdict.decision != "approve":
        return None
    final = verdict.modified or draft
    return await notes_repo.create(title=final.title, body=final.body)

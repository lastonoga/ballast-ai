"""Brainstorm workflow package.

Re-exports the public surface so external callers keep one stable
import path::

    from notes_app.workflows.brainstorm import brainstorm, workflow_id

Internals (the divergent assembly, the typed events, the prompts)
are accessible as ``notes_app.workflows.brainstorm.{events,divergent,
prompts}`` for observers / customisation.
"""

from notes_app.workflows.brainstorm.events import (
    BrainstormCancelled,
    BrainstormChose,
    BrainstormEvent,
    BrainstormSaved,
    BrainstormTimedOut,
    brainstorm_progress,
    default_chat_router,
)
from notes_app.workflows.brainstorm.flow import brainstorm, workflow_id
from notes_app.workflows.brainstorm.prompts import (
    BrainstormAgentSpec,
    DEFAULT_DIVERGENT_SPECS,
)

__all__ = [
    "BrainstormAgentSpec",
    "BrainstormCancelled",
    "BrainstormChose",
    "BrainstormEvent",
    "BrainstormSaved",
    "BrainstormTimedOut",
    "DEFAULT_DIVERGENT_SPECS",
    "brainstorm",
    "brainstorm_progress",
    "default_chat_router",
    "workflow_id",
]

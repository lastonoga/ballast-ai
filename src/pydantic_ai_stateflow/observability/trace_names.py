"""Canonical OTel/logfire span names emitted by the framework.

Using ``StrEnum`` so values remain valid strings (pass directly to
``logfire.span(name)``) while still discoverable / refactorable. One
namespace per major subsystem; flat values so dashboards can filter
with prefix queries (e.g. ``name STARTS WITH 'persistence.'``).

Add a new entry here when wiring ``@traced`` somewhere new — keep the
string consistent with the module path so logfire's tree groups them
naturally.
"""

from __future__ import annotations

from enum import StrEnum


class TraceName(StrEnum):
    """Span names emitted by ``pydantic_ai_stateflow``."""

    # --- api.streaming -----------------------------------------------------
    STREAMING_POST_MESSAGE = "api.streaming.post_message"

    # --- api.threads -------------------------------------------------------
    THREADS_CREATE = "api.threads.create"
    THREADS_GET_MESSAGES = "api.threads.get_messages"
    THREADS_ADD_MESSAGE = "api.threads.add_message"

    # --- persistence.thread ------------------------------------------------
    THREAD_CREATE = "persistence.thread.create"
    THREAD_ADD_MESSAGE = "persistence.thread.add_message"
    THREAD_HISTORY = "persistence.thread.history"

    # --- patterns ----------------------------------------------------------
    PATTERN_MAP_REDUCE = "pattern.map_reduce"
    PATTERN_REFLECTION = "pattern.reflection"
    PATTERN_MUTATION_PIPELINE = "pattern.mutation_pipeline"
    PATTERN_HITL_GATE = "pattern.hitl_gate"

    # --- hitl channels -----------------------------------------------------
    CHANNEL_UI = "channel.ui"
    CHANNEL_WEBHOOK = "channel.webhook"
    CHANNEL_CONVERSATIONAL = "channel.conversational"

    # --- capabilities ------------------------------------------------------
    CAPABILITY_PII_GUARD = "capability.pii_guard"
    CAPABILITY_PII_GUARD_STREAM = "capability.pii_guard.stream"
    CAPABILITY_BUDGET_GUARD = "capability.budget_guard"


__all__ = ["TraceName"]

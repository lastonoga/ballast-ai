"""``ThreadChannel`` — HITL channel that opens a helper sub-thread.

The helper agent's tools call ``Durable.send_async`` to push back a
typed verdict via the DBOS topic mechanism (the same as
``UICardChannel``).  Both channels share ``CardVerdict[InT]`` as the
verdict shape — there is no separate ThreadVerdict.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ballast.patterns.hitl.channels._base import DBOSHITLChannel
from ballast.patterns.hitl.channels._protocol import InT

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ballast.runtime.agents import BallastAgent


class ThreadChannel(DBOSHITLChannel[InT, "CardVerdict[InT]"]):
    """HITL channel that opens a helper sub-thread.

    The helper agent's tools call ``Durable.send_async`` to push back
    a typed verdict via the same DBOS topic mechanism as
    ``UICardChannel``.  Both channels share ``CardVerdict[InT]`` as
    the verdict shape — there is no separate ThreadVerdict.

    Constructor validates that ``helper_agent.metadata_model`` is set
    and equals ``payload_type``, so mismatches fail loud at wiring
    time rather than at runtime.
    """

    def __init__(
        self,
        *,
        helper_agent: type[BallastAgent],
        payload_type: type[InT],
        opening_message: str | None = None,
    ) -> None:
        super().__init__()
        if helper_agent.metadata_model is None:
            raise ValueError(
                f"{helper_agent.__name__}.metadata_model is None — "
                "cannot use it as a ThreadChannel helper agent.",
            )
        if helper_agent.metadata_model is not payload_type:
            raise TypeError(
                f"helper_agent.metadata_model "
                f"({helper_agent.metadata_model.__name__}) must equal "
                f"payload_type ({payload_type.__name__})",
            )
        self._helper_agent = helper_agent
        self._payload_type = payload_type
        self._opening = opening_message

    async def deliver(
        self,
        *,
        request_id: str,
        workflow_id: str,
        respond_topic: str,
        payload: InT,
    ) -> None:
        from ballast.events.context import current_parent_thread_id  # noqa: PLC0415
        from ballast.patterns.hitl.channels._thread_plumbing import (  # noqa: PLC0415
            _create_helper_thread,
            _notify_parent_thread,
            _seed_opening_message,
        )

        thread_metadata: dict[str, Any] = payload.model_dump(mode="json")
        thread_metadata["request_id"]    = request_id
        thread_metadata["workflow_id"]   = workflow_id
        thread_metadata["respond_topic"] = respond_topic

        thread_id = await _create_helper_thread(
            agent_name=self._helper_agent.name,
            metadata=thread_metadata,
        )
        if self._opening:
            await _seed_opening_message(thread_id, self._opening)

        parent = current_parent_thread_id()
        if parent is not None:
            await _notify_parent_thread(
                parent_thread_id=parent,
                helper_thread_id=thread_id,
                helper_agent_name=self._helper_agent.name,
                helper_metadata=thread_metadata,
            )

    async def decode_verdict(self, raw: Any) -> "CardVerdict[InT]":
        if raw is None:
            raise TimeoutError("ThreadChannel: verdict recv timed out")
        from pydantic import TypeAdapter  # noqa: PLC0415

        from ballast.patterns.hitl.channels.ui_card import CardVerdict  # noqa: PLC0415

        return TypeAdapter(CardVerdict[self._payload_type]).validate_python(raw)


__all__ = ["ThreadChannel"]

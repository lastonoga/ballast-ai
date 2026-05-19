"""Adapter wrapping a pydantic-ai ``Agent`` as an :class:`AgentRunner`.

Eliminates the per-app boilerplate of running ``agent.run_stream`` →
``stream_output`` → diff-against-last-emitted-prefix → emit AG-UI events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

from pydantic_ai_stateflow.api.streaming.router import (
    AgentRunner,
    StreamEvent,
    _PostMessageBody,
    extract_text,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent


OutT = TypeVar("OutT")


def _make_text_extractor(
    text_field: str | Callable[[Any], str],
) -> Callable[[Any], str]:
    if callable(text_field):
        return text_field

    attr = text_field

    def _get(out: Any) -> str:
        value = getattr(out, attr, None)
        return value or ""

    return _get


def make_runner(
    agent: Agent[Any, OutT],
    *,
    text_field: str | Callable[[OutT], str] = "reply",
    deps: Any = None,
) -> AgentRunner:
    """Wrap a pydantic-ai ``Agent`` as an :class:`AgentRunner`.

    Emits the canonical AG-UI sequence
    ``RUN_STARTED → TEXT_MESSAGE_START → TEXT_MESSAGE_CONTENT × N →
    TEXT_MESSAGE_END → RUN_FINISHED`` (and ``RUN_ERROR`` on exception).

    Args:
      agent: configured pydantic-ai Agent (typically with a structured
        ``output_type`` BaseModel).
      text_field: how to pull the streaming text out of each progressive
        output snapshot:

        - ``str`` — attribute name on the BaseModel (default ``"reply"``).
        - ``Callable[[OutT], str]`` — applied to each snapshot; must return
          the FULL text so far (the adapter does the diffing).

      deps: forwarded to ``agent.run_stream(..., deps=deps)``.

    Diffing rules (so the runner emits true deltas, not snapshots):
      - If the new snapshot extends the last emitted text → emit the suffix.
      - If pydantic partial-validation revises the prefix (the new value
        isn't a prefix-extension) → fall back to a full re-emit of the new
        value as the delta. Never emit a negative diff.
    """
    extractor = _make_text_extractor(text_field)

    async def _runner(
        *,
        thread_id: UUID,
        run_id: UUID,
        message: _PostMessageBody,
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        del tenant_id  # adapter is stateless; routers may persist if needed
        message_id = uuid4()
        prompt = extract_text(message.parts)

        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.text_message_start(message_id=message_id)

        last_emitted = ""
        try:
            async with agent.run_stream(prompt, deps=deps) as result:
                async for snapshot in result.stream_output(debounce_by=0.05):
                    current = extractor(snapshot) or ""
                    if not current:
                        continue
                    if current == last_emitted:
                        continue
                    if current.startswith(last_emitted):
                        delta = current[len(last_emitted):]
                    else:
                        # Partial validation revised the prefix (or shortened
                        # it); re-emit the full new value to avoid negative
                        # diffs. Client treats deltas as appends.
                        delta = current
                    last_emitted = current
                    if delta:
                        yield StreamEvent.text_message_content(
                            message_id=message_id, delta=delta,
                        )
        except Exception as exc:  # noqa: BLE001 — surface and re-raise
            yield StreamEvent.run_error(message=str(exc))
            raise

        yield StreamEvent.text_message_end(message_id=message_id)
        yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)

    return _runner

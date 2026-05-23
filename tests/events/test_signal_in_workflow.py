"""Repro: does a Signal handler connected in a PARENT workflow body
fire when the signal is emitted from a CHILD workflow body?

This mirrors the production setup in notes-app brainstorm:

    @Durable.workflow()
    async def brainstorm():
        sig.connect(closure_handler)  # parent body
        await child_workflow()        # child fires sig.send()
        sig.disconnect(closure_handler)

If signals don't propagate across this boundary, the user's progress
events never reach the chat. We've observed this empirically — this
test pins down WHETHER it's a framework bug.
"""
from __future__ import annotations

import pytest

from ballast.durable import Durable
from ballast.events import Signal


@pytest.mark.asyncio
async def test_signal_fires_in_child_workflow_when_connected_in_parent(
    fresh_dbos_executor: None,
) -> None:
    del fresh_dbos_executor
    sig = Signal("test_child_emit")
    captured: list[str] = []

    async def handler(sender, *, event, **_):  # noqa: ARG001
        captured.append(event)

    @Durable.workflow()
    async def child() -> None:
        await sig.send(sender="child", event="from-child")

    @Durable.workflow()
    async def parent() -> None:
        sig.connect(handler)
        try:
            await child()
        finally:
            sig.disconnect(handler)

    await parent()
    assert captured == ["from-child"], (
        f"Handler did not see child-workflow emit. Got: {captured!r}"
    )


@pytest.mark.asyncio
async def test_signal_fires_in_same_workflow_body(
    fresh_dbos_executor: None,
) -> None:
    """Baseline: handler-in-body + send-in-same-body. Must work."""
    del fresh_dbos_executor
    sig = Signal("test_same_body")
    captured: list[str] = []

    async def handler(sender, *, event, **_):  # noqa: ARG001
        captured.append(event)

    @Durable.workflow()
    async def flow() -> None:
        sig.connect(handler)
        try:
            await sig.send(sender="flow", event="from-flow")
        finally:
            sig.disconnect(handler)

    await flow()
    assert captured == ["from-flow"]

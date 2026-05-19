"""Tests for the `AgentRunner` Protocol (F4)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

from pydantic_ai_stateflow.api.streaming import AgentRunner, StreamEvent
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody


async def _ok_runner(
    *,
    thread_id: UUID,
    run_id: UUID,
    message: _PostMessageBody,
    tenant_id: UUID,
) -> AsyncIterator[StreamEvent]:
    del thread_id, run_id, message, tenant_id
    yield StreamEvent.run_finished(uuid4(), uuid4())


def test_callable_with_correct_kwargs_satisfies_protocol() -> None:
    assert isinstance(_ok_runner, AgentRunner)


def test_lambda_satisfies_protocol_at_runtime() -> None:
    # runtime_checkable Protocols only check __call__ existence at runtime.
    async def _gen(
        *,
        thread_id: UUID,
        run_id: UUID,
        message: _PostMessageBody,
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        del thread_id, run_id, message, tenant_id
        yield StreamEvent.run_finished(uuid4(), uuid4())

    assert isinstance(_gen, AgentRunner)


def test_make_runner_satisfies_protocol() -> None:
    from pydantic_ai_stateflow.api.streaming import make_runner

    class _FakeAgent:
        def run_stream(self, *_a: object, **_kw: object) -> object:
            raise NotImplementedError

    runner = make_runner(_FakeAgent())  # type: ignore[arg-type]
    assert isinstance(runner, AgentRunner)

"""``MockAgent`` / ``MockFlow`` — test doubles for stateflow tests.

``MockAgent`` is a ``StateflowAgent`` subclass backed by pydantic-ai's
``TestModel``. ``MockFlow`` is a DBOS workflow instance with a scriptable
``run`` body. Both are construction-only test doubles — apps that want
to substitute them at runtime build their own resolution layer.
"""
from __future__ import annotations

import itertools
from typing import Any, ClassVar
from uuid import uuid4

from dbos import DBOSConfiguredInstance
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel

from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.runtime.agents import StateflowAgent


# -- MockAgent --


class MockAgent(StateflowAgent):
    """A ``StateflowAgent`` backed by pydantic-ai's ``TestModel``.

    Construct via the classmethods (``with_output`` / ``with_outputs``)
    — they wrap the underlying ``TestModel`` so calls return the
    scripted output(s) without hitting a real LLM.
    """

    metadata_model: ClassVar[None] = None
    name: ClassVar[str] = "mock-agent"

    def __init__(
        self,
        *,
        output_text: str = "mock",
        outputs: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._output_text = output_text
        self._outputs = list(outputs) if outputs else None
        self._counter = itertools.count()

    @classmethod
    def with_output(cls, text: str) -> MockAgent:
        return cls(output_text=text)

    @classmethod
    def with_outputs(cls, texts: list[str]) -> MockAgent:
        if not texts:
            raise ValueError("MockAgent.with_outputs requires at least one output")
        return cls(outputs=texts)

    def build_agent(self) -> Agent[None, str]:
        # If a list of outputs was supplied, the model cycles through
        # them by index. Otherwise it always returns the single text.
        if self._outputs is None:
            return Agent(
                TestModel(custom_output_text=self._output_text),
                output_type=str,
            )
        outputs = self._outputs
        counter = self._counter

        class _Cycling(TestModel):
            async def request(self, *args: Any, **kwargs: Any) -> Any:
                idx = next(counter) % len(outputs)
                self.custom_output_text = outputs[idx]
                return await super().request(*args, **kwargs)

        return Agent(_Cycling(), output_type=str)

    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> None:
        del thread, message
        return None


# -- MockFlow --


@Durable.dbos_class()
class MockFlow(DBOSConfiguredInstance):
    """Workflow stub registered as a real DBOS workflow.

    Used via ``TestEngine.override(SomeFlow, MockFlow.returning(...))``.
    The auto-generated route runs it through ``Durable.start_workflow``,
    so ``run`` must be a real ``@Durable.workflow`` — DBOS rejects
    unregistered functions. Each instance gets a unique ``config_name``
    so DBOS recovery and per-instance state stay clean across tests.
    """

    def __init__(
        self,
        *,
        return_value: Any | None = None,
        exception: BaseException | None = None,
    ) -> None:
        super().__init__(config_name=f"mock-flow-{uuid4()}")
        self._return_value = return_value
        self._exception = exception
        # ``calls`` is observable test state — appended to via ``record_call``
        # which is a ``@Durable.step`` so DBOS can replay deterministically
        # without re-firing side effects (we just stash on ``self``).
        self.calls: list[Any] = []

    @classmethod
    def returning(cls, output: Any) -> MockFlow:
        return cls(return_value=output)

    @classmethod
    def raising(cls, exc: BaseException) -> MockFlow:
        return cls(exception=exc)

    @Durable.workflow()
    async def run(self, input: Any) -> Any:
        self.calls.append(input)
        if self._exception is not None:
            raise self._exception
        return self._return_value


__all__ = ["MockAgent", "MockFlow"]

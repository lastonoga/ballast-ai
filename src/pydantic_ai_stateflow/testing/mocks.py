"""``MockAgent`` / ``MockFlow`` — test doubles for agents and workflows.

SKELETON stubs. Full implementation lands in SP1 T6.
"""
from __future__ import annotations

from typing import Any


class MockAgent:
    """TestModel-backed StateflowAgent. SKELETON."""

    @classmethod
    def with_output(cls, text: str) -> MockAgent:
        raise NotImplementedError("MockAgent.with_output() lands in SP1 T6")

    @classmethod
    def with_outputs(cls, texts: list[str]) -> MockAgent:
        raise NotImplementedError("MockAgent.with_outputs() lands in SP1 T6")


class MockFlow:
    """Workflow stub. SKELETON."""

    @classmethod
    def returning(cls, output: Any) -> MockFlow:
        raise NotImplementedError("MockFlow.returning() lands in SP1 T6")

    @classmethod
    def raising(cls, exc: BaseException) -> MockFlow:
        raise NotImplementedError("MockFlow.raising() lands in SP1 T6")


__all__ = ["MockAgent", "MockFlow"]

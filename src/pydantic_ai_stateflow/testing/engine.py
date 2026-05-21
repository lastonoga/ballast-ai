"""``TestEngine`` — pre-wired Stateflow app for tests.

This is a SKELETON stub. Full implementation lands in SP1 T6 — see
``docs/superpowers/specs/2026-05-22-sp1-core-rewrite-design.md`` §D.
"""
from __future__ import annotations

from types import TracebackType
from typing import Any


class TestEngine:
    """Pre-wired Stateflow app for tests. See SP1 spec §D."""

    @classmethod
    def default(cls) -> TestEngine:
        """Construct a default TestEngine (in-memory repos, fresh DBOS).

        SKELETON: raises NotImplementedError until SP1 T6 fills this in.
        """
        raise NotImplementedError("TestEngine.default() lands in SP1 T6")

    def override(self, target: type, replacement: Any) -> TestEngine:
        """Override a class registration with a replacement. SKELETON."""
        raise NotImplementedError("TestEngine.override() lands in SP1 T6")

    def test_client(self) -> Any:
        """Context-manager FastAPI TestClient. SKELETON."""
        raise NotImplementedError("TestEngine.test_client() lands in SP1 T6")

    def __enter__(self) -> TestEngine:
        raise NotImplementedError("TestEngine.__enter__ lands in SP1 T6")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError("TestEngine.__exit__ lands in SP1 T6")


__all__ = ["TestEngine"]

"""Smoke tests for Ballast.fastapi() — verify routers are mounted."""
from __future__ import annotations

import pytest

import ballast
from ballast.runtime.engine import _reset_ballast_for_tests
from ballast.settings import BallastSettings


@pytest.fixture(autouse=True)
def _reset_engine() -> None:
    """Reset the process-wide engine singleton before each test.

    Prevents cross-test contamination when other test modules (e.g.
    test_streaming_router.py) install an engine singleton without cleaning up.
    """
    _reset_ballast_for_tests()


def test_approvals_router_mounted() -> None:
    app = ballast.Ballast(BallastSettings()).fastapi()
    routes = {r.path for r in app.routes}
    assert "/approvals" in routes
    assert "/approvals/{card_id}/decision" in routes
    assert "/approvals/stream" in routes

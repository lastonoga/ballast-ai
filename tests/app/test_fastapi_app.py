"""Smoke tests for Ballast.fastapi() — verify routers are mounted."""
from __future__ import annotations

import ballast
from ballast.settings import BallastSettings


def test_approvals_router_mounted() -> None:
    app = ballast.Ballast(BallastSettings()).fastapi()
    routes = {r.path for r in app.routes}
    assert "/approvals" in routes
    assert "/approvals/{card_id}/decision" in routes
    assert "/approvals/stream" in routes

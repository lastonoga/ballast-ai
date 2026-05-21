from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter

HealthCheck = Callable[[], Awaitable[bool]]


def build_health_router(
    *,
    prefix: str = "",
    checks: dict[str, HealthCheck] | None = None,
) -> APIRouter:
    """Mount `GET {prefix}/healthz` with optional per-component checks.

    `checks` map name -> async callable returning True on healthy. Failure
    flips the overall status to "degraded" with per-check error strings.
    """
    router = APIRouter(prefix=prefix)
    cs = dict(checks or {})

    @router.get("/healthz")
    async def healthz() -> dict[str, Any]:
        if not cs:
            return {"status": "ok"}
        results: dict[str, str] = {}
        overall = "ok"
        for name, fn in cs.items():
            try:
                ok = await fn()
            except Exception as exc:  # pragma: no cover - defensive
                results[name] = f"error: {exc}"
                overall = "degraded"
                continue
            results[name] = "ok" if ok else "fail"
            if not ok:
                overall = "degraded"
        return {"status": overall, "checks": results}

    return router


# ── Module-level router (SP1 T3) ─────────────────────────────────────
# Default health router with no custom checks. Apps with checks should
# pass ``health_checks=`` to ``sf.create_app`` which builds a fresh
# router via ``build_health_router(checks=...)``.

health_router = build_health_router(checks=None)

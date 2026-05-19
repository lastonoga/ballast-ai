"""CORS configuration for :meth:`Engine.fastapi_app`.

A thin, dependency-free dataclass that mirrors Starlette's
``CORSMiddleware`` constructor kwargs. Kept here (not in ``runtime``) so
that the runtime layer doesn't grow a FastAPI / Starlette import surface
beyond what :class:`Engine` already needs.

Use :meth:`CORSConfig.permissive_dev` to wire a localhost-only dev setup
in one line; supply the dataclass directly for production where the
allowed origin list is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CORSConfig:
    """Static CORS configuration handed to Starlette's ``CORSMiddleware``.

    Field semantics mirror Starlette 1:1. ``allow_origins`` is required
    when CORS is enabled — passing an empty list is allowed but yields a
    middleware that rejects every cross-origin request (matching
    Starlette's own default behavior).
    """

    allow_origins: list[str] = field(default_factory=list)
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False
    expose_headers: list[str] = field(default_factory=list)
    max_age: int = 600

    @classmethod
    def permissive_dev(cls, *, origins: list[str] | None = None) -> CORSConfig:
        """Convenience preset for local development.

        Defaults to allowing the Next.js / Vite dev origins
        (``http://localhost:3000`` and ``http://localhost:3003``) with all
        methods and headers + credentials enabled — exactly what a
        browser-side assistant-ui dev shell needs.

        DO NOT use this in production — pass an explicit origin list.
        """
        return cls(
            allow_origins=list(
                origins if origins is not None else [
                    "http://localhost:3000",
                    "http://localhost:3003",
                ],
            ),
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

"""notes-app backend — iteration 2.

Thin FastAPI app built via `Engine.fastapi_app()` exposing thread CRUD +
AG-UI streaming, backed by a single pydantic-ai agent that talks to
OpenRouter (Qwen, structured JSON output).
"""

from notes_app.main import app, build_app

__all__ = ["app", "build_app"]

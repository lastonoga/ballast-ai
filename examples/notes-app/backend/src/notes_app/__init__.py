"""notes-app backend.

Thin FastAPI app built via ``ballast.create_app()`` exposing thread CRUD +
AG-UI streaming, backed by pydantic-ai agents that talk to OpenRouter.
"""

from notes_app.main import app

__all__ = ["app"]

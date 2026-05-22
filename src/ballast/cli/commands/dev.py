"""``stateflow dev`` — uvicorn-reload wrapper."""
from __future__ import annotations

import typer

from ballast.cli.app_detect import resolve_app_ref


def dev(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(True, "--reload/--no-reload"),
    app_ref: str | None = typer.Option(None, "--app"),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Run the stateflow app under uvicorn with reload.

    Examples:

        stateflow dev
        stateflow dev --host 0.0.0.0 --port 8001
        stateflow dev --app notes_app.main:app --no-reload
    """
    ref = resolve_app_ref(app_ref)
    import uvicorn

    uvicorn.run(
        ref.import_string,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )

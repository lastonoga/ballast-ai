"""``ballast migrate`` — Alembic wrapper using SP2 settings.dbos.database_url."""
from __future__ import annotations

import sys

import typer
from alembic.config import main as alembic_main

from ballast._alembic import resolve_alembic_ini

app = typer.Typer(
    help="Run Alembic migrations.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _resolve_alembic_ini(explicit: str | None = None) -> str:
    """Resolve alembic.ini path: --alembic-ini > pyproject.toml > bundled."""
    return resolve_alembic_ini(explicit)


@app.callback()
def _root_default(
    ctx: typer.Context,
    alembic_ini: str | None = typer.Option(None, "--alembic-ini"),
) -> None:
    """Default invocation = upgrade head."""
    if ctx.invoked_subcommand is not None:
        return
    ini = _resolve_alembic_ini(alembic_ini)
    sys.argv = ["alembic", "-c", ini, "upgrade", "head"]
    alembic_main()


@app.command(name="revision")
def revision(
    message: str = typer.Option(..., "-m", "--message"),
    autogenerate: bool = typer.Option(True, "--autogenerate/--no-autogenerate"),
    alembic_ini: str | None = typer.Option(None, "--alembic-ini"),
) -> None:
    """Create a new Alembic revision."""
    ini = _resolve_alembic_ini(alembic_ini)
    args = ["alembic", "-c", ini, "revision", "-m", message]
    if autogenerate:
        args.append("--autogenerate")
    sys.argv = args
    alembic_main()

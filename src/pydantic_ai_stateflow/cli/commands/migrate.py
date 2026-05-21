"""``stateflow migrate`` — Alembic wrapper using SP2 settings.dbos.database_url."""
from __future__ import annotations

import importlib.resources
import sys
import tomllib
from pathlib import Path

import typer
from alembic.config import main as alembic_main

app = typer.Typer(
    help="Run Alembic migrations.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _resolve_alembic_ini(explicit: str | None = None) -> str:
    """Resolve alembic.ini path: --alembic-ini > pyproject.toml > bundled."""
    if explicit:
        return explicit
    # pyproject.toml [tool.stateflow] alembic_ini
    try:
        for candidate in [Path.cwd(), *Path.cwd().parents]:
            p = candidate / "pyproject.toml"
            if p.is_file():
                with p.open("rb") as f:
                    data = tomllib.load(f)
                ini = data.get("tool", {}).get("stateflow", {}).get("alembic_ini")
                if isinstance(ini, str):
                    return ini
                break
    except (tomllib.TOMLDecodeError, OSError):
        pass
    # Fall back to bundled
    pkg_path = importlib.resources.files("pydantic_ai_stateflow") / "alembic.ini"
    return str(pkg_path)


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

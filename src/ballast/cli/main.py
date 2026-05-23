"""Top-level typer app + subcommand registration."""
from __future__ import annotations

import typer

from ballast.cli.commands import db, dev, events, migrate, workflows

cli = typer.Typer(
    name="ballast",
    help="Devtools for Ballast apps.",
    no_args_is_help=True,
)
cli.command(name="dev")(dev.dev)
cli.add_typer(migrate.app, name="migrate")
cli.add_typer(db.app, name="db")
cli.add_typer(workflows.app, name="workflows")
cli.add_typer(events.app, name="events")


def main() -> None:  # pragma: no cover
    cli()


__all__ = ["cli", "main"]

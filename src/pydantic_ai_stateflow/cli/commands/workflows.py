"""``stateflow workflows ls`` — list DBOS workflows."""
from __future__ import annotations

import asyncio
import json as _json

import typer
from rich.console import Console
from rich.table import Table

from pydantic_ai_stateflow.cli.app_detect import import_app, resolve_app_ref

app = typer.Typer(help="Workflow introspection.", no_args_is_help=True)


@app.command(name="ls")
def ls(
    app_ref: str | None = typer.Option(None, "--app"),
    status: str | None = typer.Option(
        None, "--status",
        help="Filter by status (PENDING|RUNNING|SUCCESS|ERROR|CANCELLED).",
    ),
    limit: int = typer.Option(50, "--limit"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List DBOS workflows registered in the active app.

    Imports the app (running its module-level side effects including
    workflow registration), initializes DBOS read-only, queries, then
    tears down.
    """
    ref = resolve_app_ref(app_ref)
    _ = import_app(ref)  # side-effect: imports app module so DBOS sees workflows

    from dbos import DBOSConfig

    from pydantic_ai_stateflow.durable import Durable
    from pydantic_ai_stateflow.settings import get_settings

    settings = get_settings()
    db_url = settings.dbos.database_url
    if not db_url:
        raise typer.BadParameter(
            "STATEFLOW_DBOS__DATABASE_URL is not set; cannot query DBOS.",
        )
    Durable.init(DBOSConfig(name=settings.dbos.app_name, system_database_url=db_url))
    Durable.launch()
    try:
        wfs = asyncio.run(Durable.list_workflows(limit=limit, sort_desc=True))
    finally:
        Durable.destroy(destroy_registry=False)

    # Filter by status if asked.
    if status:
        wfs = [w for w in wfs if str(getattr(w, "status", "")).upper() == status.upper()]

    if as_json:
        # The list_workflows return type is dbos-specific; serialize via repr/dict.
        out = []
        for w in wfs:
            if hasattr(w, "model_dump"):
                out.append(w.model_dump(mode="json"))
            elif hasattr(w, "__dict__"):
                out.append({k: str(v) for k, v in w.__dict__.items()})
            else:
                out.append(str(w))
        typer.echo(_json.dumps(out, indent=2, default=str))
        return

    console = Console()
    table = Table(title=f"DBOS workflows (limit {limit})")
    table.add_column("workflow_id", overflow="fold")
    table.add_column("name")
    table.add_column("status")
    table.add_column("started_at")
    table.add_column("queue")

    for w in wfs:
        wid = str(getattr(w, "workflow_id", "?"))
        nm = str(getattr(w, "name", "?"))
        st = str(getattr(w, "status", "?"))
        started = str(getattr(w, "created_at", "?"))
        queue = str(getattr(w, "queue_name", "-"))
        table.add_row(wid, nm, st, started, queue)

    console.print(table)

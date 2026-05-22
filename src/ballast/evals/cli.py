"""Typer-based CLI for stateflow evals."""
from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import typer
import yaml

from ballast.evals.traces import (
    InMemoryTraceSource,
    TraceRecord,
    TraceSource,
    dataset_from_traces,
)

app = typer.Typer(name="stateflow-evals", help="Evals CLI for pydantic-ai-stateflow")


@app.callback()
def _root() -> None:
    """Evals CLI for pydantic-ai-stateflow."""


def _demo_source() -> InMemoryTraceSource:
    return InMemoryTraceSource(records=[
        TraceRecord(
            run_id=uuid4(),
            pattern="reflection",
            inputs={"x": 1},
            output={"text": "demo"},
            created_at=datetime(2026, 4, 1, tzinfo=UTC),
            outcome="success",
        ),
    ])


def _resolve_source(path: str) -> TraceSource:
    """Import ``pkg.module:factory_callable`` and call it."""
    mod_name, attr = path.split(":")
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, attr)
    return factory()  # type: ignore[no-any-return]


_PATTERN_OPT = typer.Option(..., help="Pattern name filter")
_SINCE_OPT = typer.Option(..., help="ISO date — only newer traces included")
_OUT_OPT = typer.Option(..., help="Output YAML path")
_SOURCE_OPT = typer.Option("demo", help="Source: 'demo' (built-in) or 'pkg.module:factory'")


@app.command("dataset-from-traces")
def dataset_from_traces_cmd(
    pattern: str = _PATTERN_OPT,
    since: str = _SINCE_OPT,
    out: Path = _OUT_OPT,
    source: str = _SOURCE_OPT,
) -> None:
    """Build a YAML Dataset from production traces."""
    src: TraceSource = _demo_source() if source == "demo" else _resolve_source(source)
    since_dt = datetime.fromisoformat(since)
    if since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=UTC)
    ds = asyncio.run(dataset_from_traces(src, pattern=pattern, since=since_dt))
    payload = {
        "name": ds.name,
        "cases": [c.model_dump(mode="json") for c in ds.cases],
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    typer.echo(f"wrote {len(ds.cases)} cases -> {out}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())

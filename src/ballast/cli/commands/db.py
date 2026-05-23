"""``ballast db`` — database admin: reset, backup, restore.

All three subcommands resolve the DB URL from the alembic config
(``[alembic] sqlalchemy.url`` in the resolved alembic.ini). For
sqlite URLs we operate on the file directly (delete / copy / move);
for non-sqlite URLs ``reset`` shells through alembic
(``downgrade base`` + ``upgrade head``), and ``backup`` / ``restore``
exit with a clear pointer to ``pg_dump`` / ``pg_restore`` (rather
than wrap them — your CI / cloud SQL setup almost always has a
better backup story).

Destructive ops (``reset``, ``restore``) require a typed confirmation
unless ``--yes`` is passed.
"""

from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import typer
from alembic.config import Config as AlembicConfig
from alembic.config import main as alembic_main

from ballast._alembic import resolve_alembic_ini

app = typer.Typer(
    help="Database admin: reset / backup / restore.",
    no_args_is_help=True,
)


# ── URL helpers ────────────────────────────────────────────────────────


def _resolve_db_url(alembic_ini: str | None) -> tuple[str, str]:
    """Return ``(url, ini_path)`` resolved from the alembic config."""
    ini = resolve_alembic_ini(alembic_ini)
    cfg = AlembicConfig(ini)
    url = cfg.get_main_option("sqlalchemy.url") or ""
    if not url:
        typer.echo(
            f"[ballast db] no sqlalchemy.url in {ini}; cannot proceed.",
            err=True,
        )
        raise typer.Exit(2)
    return url, ini


def _sqlite_path(url: str) -> Path | None:
    """Extract the file path from a ``sqlite[...]:///path`` URL.

    Returns ``None`` for non-sqlite URLs or ``:memory:``.
    """
    if not url.startswith("sqlite"):
        return None
    # sqlite:///path/to.db   → /path/to.db
    # sqlite+aiosqlite:///./x → ./x
    _, _, after = url.partition(":///")
    if not after or after == ":memory:":
        return None
    return Path(after)


def _confirm(message: str, *, yes: bool) -> None:
    if yes:
        return
    typer.echo(message)
    answer = typer.prompt("Type 'yes' to confirm", default="no")
    if answer.strip().lower() != "yes":
        typer.echo("Aborted.")
        raise typer.Exit(1)


# ── reset ──────────────────────────────────────────────────────────────


@app.command(name="reset")
def reset(
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip the 'type yes to confirm' prompt.",
    ),
    alembic_ini: str | None = typer.Option(None, "--alembic-ini"),
) -> None:
    """Wipe the alembic-managed schema + data, then re-run migrations.

    SQLite: deletes the file (if it exists), then runs ``alembic
    upgrade head``. Faster + simpler than ``downgrade base`` because
    SQLite doesn't reliably honour every reversible op.

    Non-SQLite (Postgres etc.): ``alembic downgrade base`` then
    ``alembic upgrade head``. Tables defined outside the alembic
    metadata are NOT touched — that's a separate concern (manual
    ``DROP SCHEMA public CASCADE`` if you really want a nuke).

    Does NOT touch the DBOS sqlite file (workflow state). Delete it
    manually if you want a fully-clean slate:

        rm /tmp/notes-app.dbos.sqlite   # or wherever your app puts it
    """
    url, ini = _resolve_db_url(alembic_ini)
    _confirm(f"This will WIPE the database at: {url}", yes=yes)

    sqlite_path = _sqlite_path(url)
    if sqlite_path is not None:
        if sqlite_path.exists():
            sqlite_path.unlink()
            typer.echo(f"[ballast db] deleted {sqlite_path}")
        else:
            typer.echo(
                f"[ballast db] {sqlite_path} not present — nothing to delete",
            )
    else:
        typer.echo("[ballast db] alembic downgrade base …")
        sys.argv = ["alembic", "-c", ini, "downgrade", "base"]
        alembic_main()

    typer.echo("[ballast db] alembic upgrade head …")
    sys.argv = ["alembic", "-c", ini, "upgrade", "head"]
    alembic_main()
    typer.echo("[ballast db] reset complete")


# ── backup ─────────────────────────────────────────────────────────────


def _default_backup_dir() -> Path:
    return Path("backups")


def _default_backup_name(src: Path) -> str:
    stem = src.stem or "ballast"
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}-{ts}.sqlite"


@app.command(name="backup")
def backup(
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help=(
            "Destination file. Default: ./backups/<dbname>-<UTC>.sqlite"
        ),
    ),
    alembic_ini: str | None = typer.Option(None, "--alembic-ini"),
) -> None:
    """Snapshot the SQLite database file to disk.

    Non-SQLite backends exit with a pointer to ``pg_dump`` (which has
    a richer snapshot story than anything we'd wrap here).
    """
    url, _ = _resolve_db_url(alembic_ini)
    sqlite_path = _sqlite_path(url)
    if sqlite_path is None:
        typer.echo(
            "[ballast db] backup currently supports SQLite URLs only.\n"
            f"  Your URL: {url}\n"
            "  For Postgres: use ``pg_dump`` (snapshot) / ``pg_restore``\n"
            "  — your CI or managed-PG service almost certainly has a\n"
            "  better backup story than anything this CLI would wrap.",
            err=True,
        )
        raise typer.Exit(2)
    if not sqlite_path.exists():
        typer.echo(
            f"[ballast db] source DB not found: {sqlite_path}",
            err=True,
        )
        raise typer.Exit(2)

    dest = out or (_default_backup_dir() / _default_backup_name(sqlite_path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    # ``copy2`` preserves mtime — convenient when sorting by recency.
    shutil.copy2(sqlite_path, dest)
    size_kb = dest.stat().st_size // 1024
    typer.echo(f"[ballast db] snapshot {sqlite_path} → {dest} ({size_kb} KiB)")


# ── restore ────────────────────────────────────────────────────────────


@app.command(name="restore")
def restore(
    src: Path = typer.Option(
        ..., "--from", "-i",
        help="Snapshot file to restore from (created by ``ballast db backup``).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip the 'type yes to confirm' prompt.",
    ),
    alembic_ini: str | None = typer.Option(None, "--alembic-ini"),
) -> None:
    """Replace the current SQLite database file with a snapshot.

    The current file (if any) is REPLACED in place. Non-SQLite
    backends exit with a pointer to ``pg_restore``.
    """
    url, _ = _resolve_db_url(alembic_ini)
    sqlite_path = _sqlite_path(url)
    if sqlite_path is None:
        typer.echo(
            "[ballast db] restore currently supports SQLite URLs only.\n"
            f"  Your URL: {url}\n"
            "  For Postgres: use ``pg_restore`` against your snapshot.",
            err=True,
        )
        raise typer.Exit(2)
    if not src.exists():
        typer.echo(f"[ballast db] snapshot not found: {src}", err=True)
        raise typer.Exit(2)

    _confirm(
        f"This will REPLACE the database at {sqlite_path} with {src}.",
        yes=yes,
    )
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, sqlite_path)
    size_kb = sqlite_path.stat().st_size // 1024
    typer.echo(
        f"[ballast db] restored {src} → {sqlite_path} ({size_kb} KiB)",
    )

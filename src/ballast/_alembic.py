"""Shared Alembic helpers — used by both the CLI ``ballast migrate`` and
the optional auto-migrate startup hook in :class:`Ballast.fastapi`.

Resolution order for ``alembic.ini``:

1. Explicit path passed by the caller (e.g. ``--alembic-ini`` on the CLI).
2. ``[tool.ballast].alembic_ini`` in the nearest ``pyproject.toml``
   walking up from the current working directory.
3. The framework's bundled ``src/ballast/alembic.ini``.
"""
from __future__ import annotations

import importlib.resources
import tomllib
from pathlib import Path


def resolve_alembic_ini(explicit: str | None = None) -> str:
    """Resolve the alembic.ini path; see module docstring for order."""
    if explicit:
        return explicit
    try:
        for candidate in [Path.cwd(), *Path.cwd().parents]:
            p = candidate / "pyproject.toml"
            if p.is_file():
                with p.open("rb") as f:
                    data = tomllib.load(f)
                ini = data.get("tool", {}).get("ballast", {}).get("alembic_ini")
                if isinstance(ini, str):
                    # Resolve relative paths against the pyproject's directory.
                    ini_path = Path(ini)
                    if not ini_path.is_absolute():
                        ini_path = candidate / ini_path
                    return str(ini_path)
                break
    except (tomllib.TOMLDecodeError, OSError):
        pass
    pkg_path = importlib.resources.files("ballast") / "alembic.ini"
    return str(pkg_path)


__all__ = ["resolve_alembic_ini"]

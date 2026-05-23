"""Resolve the user's ballast app reference."""
from __future__ import annotations

import importlib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import typer
from fastapi import FastAPI


@dataclass(frozen=True)
class AppRef:
    module: str
    attr: str

    @property
    def import_string(self) -> str:
        return f"{self.module}:{self.attr}"


def _read_pyproject_app(start: Path | None = None) -> str | None:
    """Walk up from CWD looking for pyproject.toml with [tool.ballast] app."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            with pyproject.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            return None
        tool = data.get("tool", {}).get("ballast", {})
        app = tool.get("app")
        if isinstance(app, str):
            return app
        return None
    return None


def resolve_app_ref(explicit: str | None = None) -> AppRef:
    raw = explicit or os.environ.get("BALLAST_APP") or _read_pyproject_app()
    if not raw:
        raise typer.BadParameter(
            "Could not locate the ballast app. Set BALLAST_APP="
            "'module.path:variable_name' or add\n"
            "    [tool.ballast]\n"
            "    app = \"module.path:variable_name\"\n"
            "to pyproject.toml, or pass --app explicitly.",
        )
    module, _, attr = raw.partition(":")
    if not module or not attr:
        raise typer.BadParameter(
            f"App reference {raw!r} must be 'module.path:variable_name'.",
        )
    return AppRef(module=module, attr=attr)


def import_app(ref: AppRef) -> FastAPI:
    mod = importlib.import_module(ref.module)
    return getattr(mod, ref.attr)


__all__ = ["AppRef", "import_app", "resolve_app_ref"]

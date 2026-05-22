"""CLI smoke tests via typer's CliRunner."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ballast.cli.app_detect import (
    AppRef,
    resolve_app_ref,
)
from ballast.cli.main import cli

runner = CliRunner()


def test_help_lists_subcommands() -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "dev" in out
    assert "migrate" in out
    assert "workflows" in out
    assert "events" in out


def test_dev_help() -> None:
    result = runner.invoke(cli, ["dev", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--reload" in result.stdout or "--no-reload" in result.stdout


def test_resolve_app_ref_env(monkeypatch) -> None:
    monkeypatch.setenv("BALLAST_APP", "mymod.main:app")
    ref = resolve_app_ref()
    assert ref == AppRef(module="mymod.main", attr="app")


def test_resolve_app_ref_explicit_wins(monkeypatch) -> None:
    monkeypatch.setenv("BALLAST_APP", "envmod:app")
    ref = resolve_app_ref(explicit="explicit:thing")
    assert ref.module == "explicit"
    assert ref.attr == "thing"


def test_resolve_app_ref_missing_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("BALLAST_APP", raising=False)
    monkeypatch.chdir(tmp_path)  # away from project pyproject
    with pytest.raises(Exception, match="BALLAST_APP"):
        resolve_app_ref()


def test_resolve_app_ref_malformed(monkeypatch) -> None:
    monkeypatch.setenv("BALLAST_APP", "no-colon-here")
    with pytest.raises(Exception, match="module.path:variable_name"):
        resolve_app_ref()


def test_workflows_ls_missing_app(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("BALLAST_APP", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli, ["workflows", "ls"])
    assert result.exit_code != 0

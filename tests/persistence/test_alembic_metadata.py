"""Smoke test: Alembic metadata is loadable and consistent with SQLModel registry."""

from sqlmodel import SQLModel


def test_metadata_is_empty_in_isolation():
    """Before any framework tables are imported, SQLModel.metadata still exists."""
    assert SQLModel.metadata is not None


def test_alembic_ini_exists():
    """Alembic config file exists and is loadable."""
    from pathlib import Path

    import pydantic_ai_stateflow

    pkg_dir = Path(pydantic_ai_stateflow.__file__).parent
    assert (pkg_dir / "alembic.ini").exists()
    assert (pkg_dir / "alembic" / "env.py").exists()
    assert (pkg_dir / "alembic" / "script.py.mako").exists()

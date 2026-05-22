from ballast.runtime.dbos_setup import (
    DBOSConfig,
    build_dbos_config,
)


def test_build_dbos_config_from_dsn():
    dsn = "postgresql+asyncpg://user:pass@host:5432/dbname"
    cfg = build_dbos_config(dsn)
    assert isinstance(cfg, DBOSConfig)
    assert cfg.database_url is not None
    # asyncpg-flavored URL → DBOS expects postgresql+psycopg or plain postgresql
    assert "+asyncpg" not in cfg.database_url


def test_build_dbos_config_strips_asyncpg_dialect():
    dsn = "postgresql+asyncpg://localhost/x"
    cfg = build_dbos_config(dsn)
    assert cfg.database_url == "postgresql://localhost/x"


def test_build_dbos_config_passes_app_name():
    cfg = build_dbos_config("postgresql://localhost/x", app_name="my-app")
    assert cfg.app_name == "my-app"

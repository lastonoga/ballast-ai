from pydantic_ai_stateflow import (
    Container,
    CoreProvider,
    DBOSConfig,
    DefaultContainer,
    Engine,
    EngineInvariantViolation,
    PersistenceProvider,
    ServiceProvider,
    build_dbos_config,
)


def test_runtime_classes_visible_from_top_level() -> None:
    assert Engine is not None
    assert Container is not None
    assert isinstance(DefaultContainer(), Container)
    assert callable(build_dbos_config)
    # Verify all imports are accessible
    assert CoreProvider is not None
    assert DBOSConfig is not None
    assert EngineInvariantViolation is not None
    assert PersistenceProvider is not None
    assert ServiceProvider is not None

from pydantic_ai_stateflow import (
    DBOSConfig,
    StateflowAgent,
    build_dbos_config,
    create_app,
)


def test_runtime_classes_visible_from_top_level() -> None:
    assert DBOSConfig is not None
    assert StateflowAgent is not None
    assert callable(build_dbos_config)
    assert callable(create_app)

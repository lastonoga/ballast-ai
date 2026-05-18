from pydantic_ai_stateflow.capabilities import StateflowCapability


def test_stateflow_capability_has_name_classvar() -> None:
    class FakeCap(StateflowCapability):
        name = "fake"
    assert FakeCap.name == "fake"


def test_stateflow_capability_is_abstract_capability() -> None:
    from pydantic_ai.capabilities import AbstractCapability
    assert issubclass(StateflowCapability, AbstractCapability)


def test_stateflow_capability_requires_name_attribute_in_subclass() -> None:
    class NamelessCap(StateflowCapability):
        pass
    assert NamelessCap.name == "NamelessCap"

from ballast.capabilities import BallastCapability


def test_stateflow_capability_has_name_classvar() -> None:
    class FakeCap(BallastCapability):
        name = "fake"
    assert FakeCap.name == "fake"


def test_stateflow_capability_is_abstract_capability() -> None:
    from pydantic_ai.capabilities import AbstractCapability
    assert issubclass(BallastCapability, AbstractCapability)


def test_stateflow_capability_requires_name_attribute_in_subclass() -> None:
    class NamelessCap(BallastCapability):
        pass
    assert NamelessCap.name == "NamelessCap"

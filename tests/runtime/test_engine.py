import pytest

from pydantic_ai_stateflow.runtime.container import Container, DefaultContainer  # noqa: F401
from pydantic_ai_stateflow.runtime.engine import Engine, EngineInvariantViolation
from pydantic_ai_stateflow.runtime.provider import ServiceProvider  # noqa: F401


class _Service:
    initialised: bool = False


class _Provider:
    async def register(self, container: Container) -> None:
        container.bind(_Service, lambda _: _Service())


@pytest.mark.asyncio
async def test_engine_boots_providers_in_order():
    order: list[str] = []

    class FirstProvider:
        async def register(self, c: Container) -> None:
            order.append("first")
            c.bind(int, lambda _: 1)

    class SecondProvider:
        async def register(self, c: Container) -> None:
            order.append("second")
            # Verifies it sees first's binding (no late-resolution needed)
            assert c.get(int) == 1
            c.bind(str, lambda _: "two")

    engine = Engine(providers=[FirstProvider(), SecondProvider()])
    await engine.boot()
    assert order == ["first", "second"]


@pytest.mark.asyncio
async def test_engine_container_accessible_after_boot():
    engine = Engine(providers=[_Provider()])
    await engine.boot()
    assert isinstance(engine.container.get(_Service), _Service)


@pytest.mark.asyncio
async def test_engine_runs_invariants_after_all_providers_registered():
    invariant_seen_int: int | None = None

    async def check_int_bound(c: Container) -> None:
        nonlocal invariant_seen_int
        invariant_seen_int = c.get(int)

    class IntProvider:
        async def register(self, c: Container) -> None:
            c.bind(int, lambda _: 42)

    engine = Engine(providers=[IntProvider()], invariants=[check_int_bound])
    await engine.boot()
    assert invariant_seen_int == 42


@pytest.mark.asyncio
async def test_invariant_violation_blocks_boot():
    async def always_fail(c: Container) -> None:
        raise EngineInvariantViolation("nope")

    engine = Engine(providers=[], invariants=[always_fail])
    with pytest.raises(EngineInvariantViolation):
        await engine.boot()


@pytest.mark.asyncio
async def test_boot_is_idempotent_via_same_engine_instance():
    """Calling boot twice on same Engine raises to prevent silent re-registration."""
    engine = Engine(providers=[_Provider()])
    await engine.boot()
    with pytest.raises(RuntimeError, match="already booted"):
        await engine.boot()

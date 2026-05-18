from pydantic_ai_stateflow.runtime import Det
from pydantic_ai_stateflow.runtime.container import Container


class CoreProvider:
    """Binds core framework primitives."""

    async def register(self, container: Container) -> None:
        # Bind the Det class itself so callers can look it up via container.get
        container.bind(type(Det), lambda _: Det)

from __future__ import annotations

from typing import Any, ClassVar

from pydantic_ai.capabilities import AbstractCapability


class BallastCapability(AbstractCapability[Any]):
    """Base class for framework capabilities.

    Provides:
    - `name: ClassVar[str]` — defaults to subclass __name__ if not set.
      Used in observability spans and error messages.
    """

    name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__ or cls.__dict__["name"] == "":
            cls.name = cls.__name__

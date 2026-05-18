from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel

EntityT = TypeVar("EntityT", bound=BaseModel)


class Ref(Generic[EntityT]):
    """Typed UUID reference to an Entity of type EntityT.

    - JSON / LLM layer: plain UUID string (no wrapper object).
    - Python layer:     typed object with `.id` and `.entity_type`,
                        plus `.hydrate(repo)` for materialization.
    """

    __slots__ = ("id",)
    __entity_type__: ClassVar[type[BaseModel] | None] = None

    def __init__(self, id: UUID) -> None:
        self.id = id

    @property
    def entity_type(self) -> type[BaseModel]:
        if self.__class__.__entity_type__ is None:
            raise TypeError(
                "Ref must be subscripted with an entity type, e.g. Ref[MyEntity](uuid)"
            )
        return self.__class__.__entity_type__

    def __class_getitem__(cls, item: type[BaseModel]) -> type[Ref[Any]]:
        # Each subscripted form gets a dedicated subclass that remembers the type.
        # Cache to ensure Ref[Entity] is Ref[Entity] (same class identity).
        if not hasattr(cls, "_subscript_cache"):
            cls._subscript_cache = {}  # type: ignore
        cache: dict[type, type[Ref[Any]]] = cls._subscript_cache  # type: ignore
        if item in cache:
            return cache[item]

        cls_name = f"Ref[{item.__name__}]"
        new_cls = type(cls_name, (Ref,), {"__entity_type__": item})
        cache[item] = new_cls
        return new_cls

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ref):
            return NotImplemented
        return self.id == other.id and self.entity_type is other.entity_type

    def __hash__(self) -> int:
        return hash((self.id, self.entity_type))

    def __repr__(self) -> str:
        return f"Ref[{self.entity_type.__name__}](id={self.id!r})"

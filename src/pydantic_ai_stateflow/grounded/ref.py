from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

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
        # Per-class cache (not inherited via MRO). `cls.__dict__` is a
        # `mappingproxy` (immutable view) so we cannot call `.setdefault`
        # on it directly — we install the cache via `setattr` and check
        # presence with `in cls.__dict__` (MRO-local).
        if "_subscript_cache" not in cls.__dict__:
            cls._subscript_cache = {}  # type: ignore[attr-defined]
        cache: dict[type, type[Ref[Any]]] = cls._subscript_cache  # type: ignore[attr-defined]

        if item not in cache:
            cls_name = f"Ref[{item.__name__}]"
            cache[item] = type(cls_name, (Ref,), {"__entity_type__": item})
        return cache[item]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # Validate as UUID, then wrap into Ref instance carrying entity_type.
        # Serialize as plain UUID string (no wrapper).
        def _validate(value: Any) -> Ref[Any]:
            if isinstance(value, Ref):
                return value
            uuid_value = UUID(value) if isinstance(value, str) else value
            if not isinstance(uuid_value, UUID):
                raise TypeError(f"Ref value must be UUID or UUID-string, got {type(value)}")
            return cls(uuid_value)

        def _serialize(ref: Ref[Any]) -> str:
            return str(ref.id)

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, return_schema=core_schema.str_schema()
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ref):
            return NotImplemented
        return self.id == other.id and self.entity_type is other.entity_type

    def __hash__(self) -> int:
        return hash((self.id, self.entity_type))

    def __repr__(self) -> str:
        return f"Ref[{self.entity_type.__name__}](id={self.id!r})"

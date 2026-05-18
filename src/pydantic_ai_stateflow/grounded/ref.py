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
        # Validate using Pydantic's native UUID schema, then wrap into Ref.
        # Using `no_info_after_validator_function` (instead of plain) gives us
        # proper JSON Schema generation ({"type": "string", "format": "uuid"})
        # which the LLM-facing dynamic models (Tasks 11-13) need to advertise.
        def _wrap(value: Any) -> Ref[Any]:
            # Same-type passthrough (already correctly typed).
            if isinstance(value, cls):
                return value
            # Different-type Ref → re-wrap into THIS subscripted class to enforce typing.
            # (Prevents Ref[B] silently slipping into a Ref[A] field.)
            if isinstance(value, Ref):
                return cls(value.id)
            # Bare UUID — wrap.
            return cls(value)

        def _serialize(ref: Ref[Any]) -> str:
            return str(ref.id)

        # Use union to accept either Ref instances (from Python) or UUID strings/objects (from JSON).
        # The uuid_schema validates strings/UUID objects; the is_instance_schema allows
        # already-wrapped Ref instances to pass through.
        return core_schema.no_info_after_validator_function(
            function=_wrap,
            schema=core_schema.union_schema([
                core_schema.is_instance_schema(cls),
                core_schema.is_instance_schema(Ref),
                core_schema.uuid_schema(),
            ]),
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

from uuid import UUID, uuid4

from sqlmodel import SQLModel

from ballast.persistence.thread.domain import Message, Thread, ThreadStatus


def test_thread_table_registered() -> None:
    assert "threads" in SQLModel.metadata.tables
    assert "messages" in SQLModel.metadata.tables


def test_thread_minimal_fields() -> None:
    t = Thread(agent="conversation")
    assert isinstance(t.id, UUID)
    assert t.agent == "conversation"
    assert t.metadata_ == {}
    assert t.status == ThreadStatus.OPEN


def test_thread_with_metadata() -> None:
    t = Thread(
        agent="hitl",
        metadata_={"gate_kind": "strategy_review", "wave_id": "abc"},
    )
    assert t.metadata_["gate_kind"] == "strategy_review"


def test_thread_accepts_metadata_alias_for_construction() -> None:
    """Pydantic's ``populate_by_name=True`` lets callers use ``metadata=``
    even though the Python attr is ``metadata_`` (SQLAlchemy clash)."""
    t = Thread.model_validate({"agent": "x", "metadata": {"k": 1}})
    assert t.metadata_ == {"k": 1}


def test_thread_dump_uses_alias_for_metadata() -> None:
    """API layers should ``model_dump(by_alias=True)`` so JSON says ``metadata``."""
    t = Thread(agent="x", metadata_={"k": 1})
    dumped = t.model_dump(by_alias=True, mode="json")
    assert "metadata" in dumped
    assert dumped["metadata"] == {"k": 1}


def test_message_minimal_fields() -> None:
    m = Message(
        thread_id=uuid4(),
        role="user",
        parts=[{"kind": "text", "content": "hi"}],
    )
    assert m.role == "user"
    assert m.parts == [{"kind": "text", "content": "hi"}]

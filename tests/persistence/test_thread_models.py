from uuid import UUID, uuid4

from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadPurpose
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow


def test_thread_table_registered():
    assert "threads" in SQLModel.metadata.tables
    assert "messages" in SQLModel.metadata.tables


def test_thread_purpose_enum_values():
    assert ThreadPurpose.ONBOARDING == "onboarding"
    assert ThreadPurpose.CONVERSATION == "conversation"
    assert ThreadPurpose.HITL == "hitl"


def test_thread_row_minimal_fields():
    row = ThreadRow(
        tenant_id=uuid4(),
        purpose=ThreadPurpose.CONVERSATION.value,
        actor_id="user-1",
    )
    assert isinstance(row.id, UUID)
    assert row.purpose == "conversation"
    assert row.actor_id == "user-1"
    assert row.purpose_metadata == {}


def test_thread_row_with_purpose_metadata():
    row = ThreadRow(
        tenant_id=uuid4(),
        purpose=ThreadPurpose.HITL.value,
        actor_id="founder-x",
        purpose_metadata={"gate_kind": "strategy_review", "wave_id": "abc"},
    )
    assert row.purpose_metadata["gate_kind"] == "strategy_review"


def test_message_row_fields():
    thread_id = uuid4()
    tenant_id = uuid4()
    row = MessageRow(
        tenant_id=tenant_id,
        thread_id=thread_id,
        role="user",
        parts=[{"kind": "text", "content": "hello"}],
    )
    assert row.role == "user"
    assert row.parts == [{"kind": "text", "content": "hello"}]


def test_thread_domain_from_row():
    row = ThreadRow(
        tenant_id=uuid4(),
        purpose=ThreadPurpose.CONVERSATION.value,
        actor_id="a",
    )
    domain = Thread.from_row(row)
    assert domain.id == row.id
    assert domain.purpose == ThreadPurpose.CONVERSATION
    assert domain.purpose_metadata == {}


def test_message_domain_from_row():
    row = MessageRow(
        tenant_id=uuid4(),
        thread_id=uuid4(),
        role="assistant",
        parts=[{"kind": "text", "content": "hi"}],
    )
    domain = Message.from_row(row)
    assert domain.role == "assistant"
    assert domain.parts == [{"kind": "text", "content": "hi"}]

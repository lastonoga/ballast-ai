from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence.tenant.domain import Tenant
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow


def test_tenant_row_registered_with_metadata():
    assert "tenants" in SQLModel.metadata.tables


def test_tenant_row_fields():
    row = TenantRow(id=uuid4(), name="acme")
    assert isinstance(row.id, UUID)
    assert row.name == "acme"
    assert isinstance(row.created_at, datetime)


def test_tenant_domain_roundtrip_from_row():
    row = TenantRow(id=uuid4(), name="acme")
    domain = Tenant.from_row(row)
    assert domain.id == row.id
    assert domain.name == row.name

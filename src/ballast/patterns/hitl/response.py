from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _BaseResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    actor_id: str | None = None
    answered_at: datetime
    helper_verdict: dict[str, Any] | None = None


class ApprovedResponse(_BaseResponse):
    kind: Literal["approved"] = "approved"
    feedback: str | None = None


class RejectedResponse(_BaseResponse):
    kind: Literal["rejected"] = "rejected"
    feedback: str | None = None


class ModifiedResponse(_BaseResponse):
    kind: Literal["modified"] = "modified"
    modified_proposal: dict[str, Any]
    feedback: str | None = None


class TimeoutResponse(_BaseResponse):
    kind: Literal["timeout"] = "timeout"


HITLResponse = Annotated[
    ApprovedResponse | RejectedResponse | ModifiedResponse | TimeoutResponse,
    Field(discriminator="kind"),
]

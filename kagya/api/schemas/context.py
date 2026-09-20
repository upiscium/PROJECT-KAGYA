"""Context administration API schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from kagya.identifiers import validate_identifier


class ContextRelationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    related_context_id: StrictStr

    @field_validator("related_context_id")
    @classmethod
    def validate_related_context_id(cls, value: str) -> str:
        try:
            return validate_identifier(value)
        except Exception:
            raise ValueError("Invalid request") from None


class ContextFrameResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_id: str
    context_type: str
    source_channel: str
    source_session_id: str | None
    participant_refs: tuple[str, ...]
    parent_context_id: str | None
    related_context_ids: tuple[str, ...]
    status: str
    created_revision: int
    last_modified_revision: int
    started_at: datetime
    last_active_at: datetime
    is_current: bool


class ContextListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_context_id: str | None
    contexts: tuple[ContextFrameResponse, ...]


class ContextRelationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contexts: tuple[ContextFrameResponse, ...]

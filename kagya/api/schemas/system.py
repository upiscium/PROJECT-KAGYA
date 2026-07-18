"""System metadata API schemas."""

from pydantic import BaseModel


class BuildInfoSchema(BaseModel):
    version: str
    commit: str | None = None


class RuntimeInfoSchema(BaseModel):
    environment: str
    provider: str
    primary_model_id: str
    fallback_configured: bool
    transformers_4bit: bool
    qlora_dry_run: bool
    admin_token_configured: bool


class SystemInfoResponse(BaseModel):
    project: str
    status: str
    build: BuildInfoSchema
    runtime: RuntimeInfoSchema


class RuntimeEventSchema(BaseModel):
    id: int
    timestamp: str
    category: str
    event_type: str
    message: str
    metadata: dict[str, object]


class RuntimeEventListResponse(BaseModel):
    events: list[RuntimeEventSchema]


class JournalRecordSchema(BaseModel):
    record_id: str
    timestamp: str
    lifecycle: str
    event_id: str
    event_type: str
    source: str
    processing_sequence: int | None
    snapshot_sequence: int | None
    causation_id: str | None
    correlation_id: str | None
    state_hash_before: str | None
    state_hash_after: str | None
    snapshot_hash: str | None
    failure_category: str | None
    actor_id: str | None
    actor_role: str | None
    target: str | None
    reauthenticated: bool | None
    previous_record_hash: str | None
    record_hash: str


class JournalRecordListResponse(BaseModel):
    records: list[JournalRecordSchema]

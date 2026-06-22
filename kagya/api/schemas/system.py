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

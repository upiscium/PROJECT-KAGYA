"""Memory API schemas."""

from typing import Any

from pydantic import BaseModel


class MemoryMetadataUpdateRequest(BaseModel):
    tags: list[str] | None = None
    operator_metadata: dict[str, Any] | None = None


class MemoryReviewRequest(BaseModel):
    validation_status: str
    lifecycle_status: str


class EpisodeMemoryResponse(BaseModel):
    id: str
    user_input: str
    response: str
    loss: float
    emotion_valence: float
    emotion_arousal: float
    record_type: str
    archived: bool
    created_at: str
    tags: list[str]
    operator_metadata: dict[str, Any]
    validation_status: str
    lifecycle_status: str
    generation_healthy: bool
    generation_health_reasons: list[str]
    content_hash: str
    source_event_id: str | None
    source: str
    processing_sequence: int | None
    provider: str
    model_id: str
    model_revision: str
    adapter_id: str | None
    consolidation_status: str


class SemanticMemoryResponse(BaseModel):
    id: str
    text: str
    source_episode_ids: list[str]
    record_type: str
    archived: bool
    created_at: str
    tags: list[str]
    operator_metadata: dict[str, Any]


class MemorySearchResponse(BaseModel):
    db1_results: list[EpisodeMemoryResponse]
    db2_results: list[SemanticMemoryResponse]

"""Memory API schemas."""

from typing import Any

from pydantic import BaseModel


class MemoryMetadataUpdateRequest(BaseModel):
    tags: list[str] | None = None
    operator_metadata: dict[str, Any] | None = None


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

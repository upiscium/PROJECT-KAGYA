"""Memory API schemas."""

from pydantic import BaseModel


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


class SemanticMemoryResponse(BaseModel):
    id: str
    text: str
    source_episode_ids: list[str]
    record_type: str
    created_at: str


class MemorySearchResponse(BaseModel):
    db1_results: list[EpisodeMemoryResponse]
    db2_results: list[SemanticMemoryResponse]

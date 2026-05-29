"""Chat API schemas."""

from pydantic import BaseModel, Field


class AttachmentSchema(BaseModel):
    type: str = Field(min_length=1)
    url: str | None = None
    name: str | None = None
    content_type: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    attachments: list[AttachmentSchema] = Field(default_factory=list)
    debug: bool = False


class EmotionSchema(BaseModel):
    valence: float
    arousal: float
    optimal_loss: float


class ModelSchema(BaseModel):
    model_id: str
    adapter_id: str | None = None


class ChatResponse(BaseModel):
    episode_id: str
    response: str
    emotion: EmotionSchema
    model: ModelSchema

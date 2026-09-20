"""Chat API schemas."""

from pydantic import BaseModel, Field, StrictStr, field_validator

from kagya.identifiers import validate_identifier


class AttachmentSchema(BaseModel):
    type: str = Field(min_length=1)
    url: str | None = None
    name: str | None = None
    content_type: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    attachments: list[AttachmentSchema] = Field(default_factory=list)
    debug: bool = False
    context_id: StrictStr | None = None
    client_session_id: StrictStr | None = None
    interlocutor_key: StrictStr | None = None

    @field_validator("context_id", "client_session_id", "interlocutor_key")
    @classmethod
    def validate_selector(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_identifier(value)
        except Exception:
            raise ValueError("selector must be a valid opaque identifier") from None


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

"""Chat API schemas."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from kagya.operation_status import OperationStatus


class AttachmentSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1)
    url: str | None = None
    name: str | None = None
    content_type: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    text: str = Field(min_length=1, validation_alias=AliasChoices("text", "message"))
    attachments: list[AttachmentSchema] = Field(default_factory=list)
    debug: bool = False
    context_id: str | None = None
    client_session_id: str | None = None
    interlocutor_key: str | None = None


class EmotionSchema(BaseModel):
    valence: float
    arousal: float
    optimal_loss: float


class ModelSchema(BaseModel):
    model_id: str
    adapter_id: str | None = None
    adapter_hash: str | None = None
    activation_sequence: int | None = None
    fallback_used: bool = False


class ChatResponse(BaseModel):
    context_id: str
    episode_id: str
    experience_id: str
    response: str
    emotion: EmotionSchema
    model: ModelSchema


class ChatJobAccepted(BaseModel):
    operation: OperationStatus
    status_url: str
    result_url: str
    events_url: str
    duplicate: bool = False


class ChatJobStatus(BaseModel):
    operation: OperationStatus


class ChatJobResult(BaseModel):
    operation: OperationStatus
    result: ChatResponse


class ChatCancelResponse(BaseModel):
    disposition: str
    operation: OperationStatus

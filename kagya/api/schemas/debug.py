"""Debug API schemas."""

from pydantic import BaseModel

from kagya.api.schemas.chat import ChatResponse, EmotionSchema
from kagya.body import EmotionUpdateReasonCode
from kagya.cognition import AppraisalReasonCode, LossInvalidReason


class RetrievedEpisodeSchema(BaseModel):
    id: str
    user_input: str
    response: str
    record_type: str


class RetrievedSemanticSchema(BaseModel):
    id: str
    text: str
    record_type: str


class RetrievedMemorySchema(BaseModel):
    db1_results: list[RetrievedEpisodeSchema]
    db2_results: list[RetrievedSemanticSchema]


class GenerationParamsSchema(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


class LossMeasurementSchema(BaseModel):
    model_key: str
    raw_loss: float | None
    valid: bool
    invalid_reason: LossInvalidReason | None
    calibrated_novelty: float | None


class AppraisalSchema(BaseModel):
    novelty: float | None
    novelty_valid: bool
    goal_progress: float | None
    threat: float | None
    controllability: float | None
    certainty: float | None
    social_relevance: float | None
    effort_cost: float | None
    reasons: list[AppraisalReasonCode]


class ValenceContributionsSchema(BaseModel):
    goal_progress: float
    threat: float
    effort_cost: float
    controllability: float


class ArousalContributionsSchema(BaseModel):
    novelty: float
    threat: float
    effort_cost: float
    social_relevance: float
    uncertainty: float
    low_controllability: float


class EmotionUpdateSchema(BaseModel):
    state: EmotionSchema
    valence_contributions: ValenceContributionsSchema
    arousal_contributions: ArousalContributionsSchema
    reasons: list[EmotionUpdateReasonCode]


class ChatDiagnosticsSchema(BaseModel):
    measurement: LossMeasurementSchema
    appraisal: AppraisalSchema
    temporal_update: EmotionUpdateSchema
    emotion_update: EmotionUpdateSchema


class DebugChatResponse(ChatResponse):
    hidden_thought: str
    loss: float | None
    prompt: str
    retrieved_memory: RetrievedMemorySchema
    generation_params: GenerationParamsSchema
    diagnostics: ChatDiagnosticsSchema


class EmotionStateResponse(EmotionSchema):
    pass

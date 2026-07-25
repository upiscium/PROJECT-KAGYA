"""Debug API schemas."""

from pydantic import BaseModel

from kagya.structured_response import PublicBehaviorClass, StructuredResponseStatus

from kagya.api.schemas.chat import AttachmentSchema, ChatResponse, EmotionSchema


class RetrievedEpisodeSchema(BaseModel):
    id: str
    user_input: str
    response: str
    record_type: str
    context_id: str | None
    semantic_relevance: float
    context_compatibility: float
    context_relation: str
    cross_context: bool


class RetrievedSemanticSchema(BaseModel):
    id: str
    text: str
    record_type: str
    context_id: str | None
    semantic_relevance: float
    context_compatibility: float
    context_relation: str
    cross_context: bool


class RetrievedMemorySchema(BaseModel):
    db1_results: list[RetrievedEpisodeSchema]
    db2_results: list[RetrievedSemanticSchema]


class GenerationParamsSchema(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float
    no_repeat_ngram_size: int


class WorkingMemoryDecisionSchema(BaseModel):
    item_id: str
    kind: str
    selected: bool
    score: float
    reasons: list[str]
    activation: float
    salience: float
    retention_reason: str
    reference: str | None
    context_id: str | None
    context_compatibility: float
    context_relation: str
    cross_context: bool


class WorkingMemoryViewSchema(BaseModel):
    items: list[WorkingMemoryDecisionSchema]
    token_count: int
    item_capacity: int
    token_capacity: int


class LossMeasurementSchema(BaseModel):
    raw_loss: float | None
    mean_token_loss: float | None
    target_token_count: int | None
    model_key: str
    valid: bool
    invalid_reason: str | None
    calibrated_novelty: float | None


class AppraisalSchema(BaseModel):
    novelty: float | None
    goal_progress: float
    threat: float
    controllability: float
    certainty: float
    social_relevance: float
    effort_cost: float
    novelty_valid: bool
    reasons: list[str]


class EmotionUpdateSchema(BaseModel):
    valence_contributions: dict[str, float]
    arousal_contributions: dict[str, float]
    reasons: list[str]


class DebugChatResponse(ChatResponse):
    hidden_thought: str
    behavior_class: PublicBehaviorClass
    response_parse_valid: bool
    response_status: StructuredResponseStatus
    loss: float | None
    prompt: str
    attachments: list[AttachmentSchema]
    retrieved_memory: RetrievedMemorySchema
    generation_params: GenerationParamsSchema
    working_memory: WorkingMemoryViewSchema
    loss_measurement: LossMeasurementSchema
    appraisal: AppraisalSchema
    emotion_update: EmotionUpdateSchema


class EmotionStateResponse(EmotionSchema):
    pass

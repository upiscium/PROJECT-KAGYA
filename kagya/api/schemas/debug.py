"""Debug API schemas."""

from pydantic import BaseModel

from kagya.api.schemas.chat import AttachmentSchema, ChatResponse, EmotionSchema


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
    repetition_penalty: float
    no_repeat_ngram_size: int


class DebugChatResponse(ChatResponse):
    hidden_thought: str
    loss: float
    prompt: str
    attachments: list[AttachmentSchema]
    retrieved_memory: RetrievedMemorySchema
    generation_params: GenerationParamsSchema


class EmotionStateResponse(EmotionSchema):
    pass

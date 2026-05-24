"""Development-only debug routes."""

from fastapi import APIRouter, Depends

from kagya.api.dependencies import get_api_settings, get_main_loop
from kagya.api.routes.chat import chat_response_from_result
from kagya.api.schemas.chat import ChatRequest
from kagya.api.schemas.debug import (
    DebugChatResponse,
    EmotionStateResponse,
    GenerationParamsSchema,
    RetrievedEpisodeSchema,
    RetrievedMemorySchema,
    RetrievedSemanticSchema,
)
from kagya.config import Settings
from kagya.runtime import KagyaMainLoop


router = APIRouter(prefix="/api", tags=["debug"])


@router.post("/chat/debug", response_model=DebugChatResponse)
def debug_chat(
    request: ChatRequest,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    settings: Settings = Depends(get_api_settings),
) -> DebugChatResponse:
    """Development-only debug chat. Future auth must gate this endpoint."""

    result = main_loop.chat(request.message, debug=True)
    base = chat_response_from_result(result)
    return DebugChatResponse(
        **base.model_dump(),
        hidden_thought=result.hidden_thought,
        loss=result.loss,
        prompt=result.prompt,
        retrieved_memory=RetrievedMemorySchema(
            db1_results=[
                RetrievedEpisodeSchema(
                    id=record.id,
                    user_input=record.user_input,
                    response=record.response,
                    record_type=record.record_type.value,
                )
                for record in result.memory_context.db1_results
            ],
            db2_results=[
                RetrievedSemanticSchema(
                    id=record.id,
                    text=record.text,
                    record_type=record.record_type.value,
                )
                for record in result.memory_context.db2_results
            ],
        ),
        generation_params=GenerationParamsSchema(
            max_new_tokens=settings.generation.max_new_tokens,
            temperature=settings.generation.temperature,
            top_p=settings.generation.top_p,
            do_sample=settings.generation.do_sample,
        ),
    )


@router.get("/state/emotion", response_model=EmotionStateResponse)
def emotion_state(main_loop: KagyaMainLoop = Depends(get_main_loop)) -> EmotionStateResponse:
    state = main_loop.emotion_engine.state
    return EmotionStateResponse(
        valence=state.valence,
        arousal=state.arousal,
        optimal_loss=state.optimal_loss,
    )

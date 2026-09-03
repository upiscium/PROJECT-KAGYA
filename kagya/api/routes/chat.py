"""Chat routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import get_agent_runtime, get_main_loop
from kagya.api.runtime_execution import execute
from kagya.api.schemas.chat import ChatRequest, ChatResponse, EmotionSchema, ModelSchema
from kagya.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    ChatResult,
    KagyaMainLoop,
)


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatResponse:
    reject_unsupported_attachments(request)
    result = execute(
        runtime,
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: main_loop.chat(request.message),
    )
    return chat_response_from_result(result)


def reject_unsupported_attachments(request: ChatRequest) -> None:
    if request.attachments:
        raise HTTPException(
            status_code=422,
            detail="Attachments are schema-only in v1.0; runtime execution is text-only.",
        )


def chat_response_from_result(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        episode_id=result.episode_id,
        response=result.response,
        emotion=EmotionSchema(
            valence=result.valence,
            arousal=result.arousal,
            optimal_loss=result.optimal_loss,
        ),
        model=ModelSchema(model_id=result.model_id, adapter_id=result.adapter_id),
    )

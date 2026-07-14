"""Chat routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    get_runtime_event_log,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.schemas.chat import ChatRequest, ChatResponse, EmotionSchema, ModelSchema
from kagya.runtime import AgentEventType, AgentRuntime, ChatResult


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    http_request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> ChatResponse:
    try:
        result = execute_agent_event(
            runtime,
            AgentEventType.CHAT,
            source="api.chat",
            handler=lambda: get_main_loop(http_request).chat(
                request.text, debug=False, attachments=attachment_metadata(request)
            ),
            payload={
                "text": request.text,
                "attachments": attachment_metadata(request),
            },
        ).value
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result.fallback_used:
        event_log.record(
            category="model",
            event_type="fallback_used",
            message="Chat response used the fallback model",
            metadata={
                "model_id": result.model_id,
                "adapter_id": result.adapter_id,
                "debug": False,
            },
        )
    return chat_response_from_result(result)


def chat_response_from_result(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        episode_id=result.episode_id,
        response=result.response,
        emotion=EmotionSchema(
            valence=result.valence,
            arousal=result.arousal,
            optimal_loss=result.optimal_loss,
        ),
        model=ModelSchema(
            model_id=result.model_id,
            adapter_id=result.adapter_id,
            fallback_used=result.fallback_used,
        ),
    )


def attachment_metadata(request: ChatRequest) -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in attachment.model_dump(
                include={"type", "name", "url", "content_type"}
            ).items()
            if value is not None
        }
        for attachment in request.attachments
    ]

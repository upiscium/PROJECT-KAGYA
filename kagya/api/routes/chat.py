"""Chat routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import get_main_loop
from kagya.api.schemas.chat import ChatRequest, ChatResponse, EmotionSchema, ModelSchema
from kagya.runtime import ChatResult, KagyaMainLoop


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, main_loop: KagyaMainLoop = Depends(get_main_loop)
) -> ChatResponse:
    try:
        result = main_loop.chat(
            request.text, debug=False, attachments=attachment_metadata(request)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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

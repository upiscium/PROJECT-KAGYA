"""Chat routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from uuid import uuid4

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    get_runtime_event_log,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.schemas.chat import ChatRequest, ChatResponse, EmotionSchema, ModelSchema
from kagya.runtime import AgentEventType, AgentRuntime, ChatResult
from kagya.identity import OriginActor


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    http_request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> ChatResponse:
    context_id = request.context_id or f"ctx-{uuid4()}"
    try:
        result = execute_agent_event(
            runtime,
            AgentEventType.CHAT,
            source="api.chat",
            handler=lambda: get_main_loop(http_request).chat(
                request.text,
                debug=False,
                attachments=attachment_metadata(request),
                context_id=context_id,
                source_channel="api.chat",
                source_session_id=request.client_session_id,
                interlocutor_key=request.interlocutor_key,
                create_context=request.context_id is None,
                origin_actor=OriginActor.USER,
            ),
            payload={
                "attachments": attachment_metadata(request),
            },
            correlation_id=context_id,
        ).value
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Context not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    record_appraisal(event_log, result, debug=False)
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
        context_id=result.context_id,
        episode_id=result.episode_id,
        experience_id=result.experience_id,
        response=result.response,
        emotion=EmotionSchema(
            valence=result.valence,
            arousal=result.arousal,
            optimal_loss=result.optimal_loss,
        ),
        model=ModelSchema(
            model_id=result.model_id,
            adapter_id=result.adapter_id,
            adapter_hash=result.adapter_hash,
            activation_sequence=result.activation_sequence,
            fallback_used=result.fallback_used,
        ),
    )


def record_appraisal(
    event_log: RuntimeEventLog, result: ChatResult, *, debug: bool
) -> None:
    event_log.record(
        category="appraisal",
        event_type="evaluated"
        if result.loss_measurement.valid
        else "measurement_invalid",
        message="Cognitive appraisal completed",
        metadata={
            "model_key": result.loss_measurement.model_key,
            "measurement_valid": result.loss_measurement.valid,
            "invalid_reason": result.loss_measurement.invalid_reason,
            "novelty": result.appraisal.novelty,
            "goal_progress": result.appraisal.goal_progress,
            "threat": result.appraisal.threat,
            "controllability": result.appraisal.controllability,
            "certainty": result.appraisal.certainty,
            "social_relevance": result.appraisal.social_relevance,
            "effort_cost": result.appraisal.effort_cost,
            "reasons": list(result.emotion_update.reasons),
            "debug": debug,
        },
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

"""Development-only debug routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from uuid import uuid4

from kagya.api.dependencies import (
    get_api_settings,
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    get_runtime_event_log,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.routes.chat import (
    attachment_metadata,
    chat_response_from_result,
    record_appraisal,
)
from kagya.api.schemas.chat import ChatRequest
from kagya.api.schemas.debug import (
    DebugChatResponse,
    EmotionStateResponse,
    GenerationParamsSchema,
    RetrievedEpisodeSchema,
    RetrievedMemorySchema,
    RetrievedSemanticSchema,
    WorkingMemoryDecisionSchema,
    WorkingMemoryViewSchema,
    AppraisalSchema,
    EmotionUpdateSchema,
    LossMeasurementSchema,
)
from kagya.config import Settings
from kagya.runtime import AgentEventType, AgentRuntime
from kagya.identity import OriginActor


router = APIRouter(prefix="/api", tags=["debug"])


@router.post("/chat/debug", response_model=DebugChatResponse)
def debug_chat(
    request: ChatRequest,
    http_request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> DebugChatResponse:
    """Development-only debug chat gated by explicit opt-in."""

    if not request.debug:
        raise HTTPException(status_code=400, detail="Debug access requires debug=true")

    context_id = request.context_id or f"ctx-{uuid4()}"
    try:
        result = execute_agent_event(
            runtime,
            AgentEventType.DEBUG_CHAT,
            source="api.chat.debug",
            handler=lambda: get_main_loop(http_request).chat(
                request.text,
                debug=True,
                attachments=attachment_metadata(request),
                context_id=context_id,
                source_channel="api.chat.debug",
                source_session_id=request.client_session_id,
                interlocutor_key=request.interlocutor_key,
                create_context=request.context_id is None,
                origin_actor=OriginActor.OPERATOR,
            ),
            payload={
                "text": request.text,
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
    record_appraisal(event_log, result, debug=True)
    if result.fallback_used:
        event_log.record(
            category="model",
            event_type="fallback_used",
            message="Debug chat response used the fallback model",
            metadata={
                "model_id": result.model_id,
                "adapter_id": result.adapter_id,
                "debug": True,
            },
        )
    base = chat_response_from_result(result)
    return DebugChatResponse(
        **base.model_dump(),
        hidden_thought=result.hidden_thought,
        behavior_class=result.behavior_class,
        response_parse_valid=result.response_parse_valid,
        response_status=result.response_status,
        loss=result.loss,
        prompt=result.prompt,
        attachments=request.attachments,
        retrieved_memory=RetrievedMemorySchema(
            db1_results=[
                RetrievedEpisodeSchema(
                    id=record.id,
                    user_input=record.user_input,
                    response=record.response,
                    record_type=record.record_type.value,
                    context_id=record.context_id,
                    semantic_relevance=record.semantic_relevance,
                    context_compatibility=record.context_compatibility,
                    context_relation=record.context_relation,
                    cross_context=record.cross_context,
                )
                for record in result.memory_context.db1_results
            ],
            db2_results=[
                RetrievedSemanticSchema(
                    id=record.id,
                    text=record.text,
                    record_type=record.record_type.value,
                    context_id=record.context_id,
                    semantic_relevance=record.semantic_relevance,
                    context_compatibility=record.context_compatibility,
                    context_relation=record.context_relation,
                    cross_context=record.cross_context,
                )
                for record in result.memory_context.db2_results
            ],
        ),
        generation_params=GenerationParamsSchema(
            max_new_tokens=settings.generation.max_new_tokens,
            temperature=settings.generation.temperature,
            top_p=settings.generation.top_p,
            do_sample=settings.generation.do_sample,
            repetition_penalty=settings.generation.repetition_penalty,
            no_repeat_ngram_size=settings.generation.no_repeat_ngram_size,
        ),
        working_memory=WorkingMemoryViewSchema(
            items=[
                WorkingMemoryDecisionSchema(
                    item_id=decision.item_id,
                    kind=decision.kind.value,
                    selected=decision.selected,
                    score=decision.score,
                    reasons=list(decision.reasons),
                    activation=decision.activation,
                    salience=decision.salience,
                    retention_reason=decision.retention_reason.value,
                    reference=decision.reference,
                    context_id=decision.context_id,
                    context_compatibility=decision.context_compatibility,
                    context_relation=decision.context_relation,
                    cross_context=decision.cross_context,
                )
                for decision in result.working_memory_view.decisions
            ],
            token_count=result.working_memory_view.token_count,
            item_capacity=result.working_memory_view.item_capacity,
            token_capacity=result.working_memory_view.token_capacity,
        ),
        loss_measurement=LossMeasurementSchema(
            raw_loss=result.loss_measurement.raw_loss,
            mean_token_loss=result.loss_measurement.mean_token_loss,
            target_token_count=result.loss_measurement.target_token_count,
            model_key=result.loss_measurement.model_key,
            valid=result.loss_measurement.valid,
            invalid_reason=result.loss_measurement.invalid_reason,
            calibrated_novelty=result.loss_measurement.calibrated_novelty,
        ),
        appraisal=AppraisalSchema(
            novelty=result.appraisal.novelty,
            goal_progress=result.appraisal.goal_progress,
            threat=result.appraisal.threat,
            controllability=result.appraisal.controllability,
            certainty=result.appraisal.certainty,
            social_relevance=result.appraisal.social_relevance,
            effort_cost=result.appraisal.effort_cost,
            novelty_valid=result.appraisal.novelty_valid,
            reasons=list(result.appraisal.reasons),
        ),
        emotion_update=EmotionUpdateSchema(
            valence_contributions=result.emotion_update.valence_contributions,
            arousal_contributions=result.emotion_update.arousal_contributions,
            reasons=list(result.emotion_update.reasons),
        ),
    )


@router.get("/state/emotion", response_model=EmotionStateResponse)
def emotion_state(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> EmotionStateResponse:
    state = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_READ,
        source="api.state.emotion",
        handler=lambda: get_main_loop(request).emotion_engine.state,
    ).value
    return EmotionStateResponse(
        valence=state.valence,
        arousal=state.arousal,
        optimal_loss=state.optimal_loss,
    )

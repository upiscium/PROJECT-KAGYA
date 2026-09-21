"""Development-only debug routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import (
    get_agent_runtime,
    get_api_settings,
    get_main_loop,
    require_admin,
)
from kagya.api.runtime_execution import execute
from kagya.api.routes.chat import (
    chat_response_from_result,
    reject_unsupported_attachments,
)
from kagya.api.schemas.chat import ChatRequest
from kagya.api.schemas.debug import (
    AppraisalSchema,
    ArousalContributionsSchema,
    ChatDiagnosticsSchema,
    DebugChatResponse,
    EmotionStateResponse,
    EmotionUpdateSchema,
    GenerationParamsSchema,
    LossMeasurementSchema,
    RetrievedEpisodeSchema,
    RetrievedMemorySchema,
    RetrievedSemanticSchema,
    ValenceContributionsSchema,
)
from kagya.body import EmotionUpdate
from kagya.cognition import AppraisalResult, LossMeasurement
from kagya.config import Settings
from kagya.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    ChatContextSelectors,
    KagyaMainLoop,
)
from kagya.runtime.main_loop import ChatDiagnostics


router = APIRouter(prefix="/api", tags=["debug"], dependencies=[Depends(require_admin)])


@router.post("/chat/debug", response_model=DebugChatResponse)
def debug_chat(
    request: ChatRequest,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    settings: Settings = Depends(get_api_settings),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> DebugChatResponse:
    """Return request-scoped diagnostics behind admin auth and explicit opt-in."""

    if not request.debug:
        raise HTTPException(status_code=400, detail="Debug access requires debug=true")
    reject_unsupported_attachments(request)
    selectors = ChatContextSelectors(
        context_id=request.context_id,
        client_session_id=request.client_session_id,
        interlocutor_key=request.interlocutor_key,
    )
    result, trace = execute(
        runtime,
        AgentEventType.DEBUG_CHAT,
        AgentEventSource.API_CHAT_DEBUG,
        lambda: main_loop.chat_debug(request.message, selectors=selectors),
    )
    base = chat_response_from_result(result)
    return DebugChatResponse(
        **base.model_dump(),
        hidden_thought=trace.hidden_thought,
        loss=result.loss,
        prompt=trace.prompt,
        retrieved_memory=RetrievedMemorySchema(
            db1_results=[
                RetrievedEpisodeSchema(
                    id=record.id,
                    user_input=record.user_input,
                    response=record.response,
                    record_type=record.record_type.value,
                )
                for record in trace.memory_context.db1_results
            ],
            db2_results=[
                RetrievedSemanticSchema(
                    id=record.id,
                    text=record.text,
                    record_type=record.record_type.value,
                )
                for record in trace.memory_context.db2_results
            ],
        ),
        generation_params=GenerationParamsSchema(
            max_new_tokens=settings.generation.max_new_tokens,
            temperature=settings.generation.temperature,
            top_p=settings.generation.top_p,
            do_sample=settings.generation.do_sample,
        ),
        diagnostics=_diagnostics_response(trace.diagnostics),
    )


def _diagnostics_response(diagnostics: ChatDiagnostics) -> ChatDiagnosticsSchema:
    return ChatDiagnosticsSchema(
        measurement=_measurement_response(diagnostics.measurement),
        appraisal=_appraisal_response(diagnostics.appraisal),
        temporal_update=_emotion_update_response(diagnostics.temporal_update),
        emotion_update=_emotion_update_response(diagnostics.emotion_update),
    )


def _measurement_response(measurement: LossMeasurement) -> LossMeasurementSchema:
    return LossMeasurementSchema(
        model_key=measurement.model_key,
        raw_loss=measurement.raw_loss,
        valid=measurement.valid,
        invalid_reason=measurement.invalid_reason,
        calibrated_novelty=measurement.calibrated_novelty,
    )


def _appraisal_response(appraisal: AppraisalResult) -> AppraisalSchema:
    return AppraisalSchema(
        novelty=appraisal.novelty,
        novelty_valid=appraisal.novelty_valid,
        goal_progress=appraisal.goal_progress,
        threat=appraisal.threat,
        controllability=appraisal.controllability,
        certainty=appraisal.certainty,
        social_relevance=appraisal.social_relevance,
        effort_cost=appraisal.effort_cost,
        reasons=list(appraisal.reasons),
    )


def _emotion_update_response(update: EmotionUpdate) -> EmotionUpdateSchema:
    return EmotionUpdateSchema(
        state=EmotionStateResponse(
            valence=update.state.valence,
            arousal=update.state.arousal,
            optimal_loss=update.state.optimal_loss,
        ),
        valence_contributions=ValenceContributionsSchema(
            goal_progress=update.valence_contributions.goal_progress,
            threat=update.valence_contributions.threat,
            effort_cost=update.valence_contributions.effort_cost,
            controllability=update.valence_contributions.controllability,
        ),
        arousal_contributions=ArousalContributionsSchema(
            novelty=update.arousal_contributions.novelty,
            threat=update.arousal_contributions.threat,
            effort_cost=update.arousal_contributions.effort_cost,
            social_relevance=update.arousal_contributions.social_relevance,
            uncertainty=update.arousal_contributions.uncertainty,
            low_controllability=update.arousal_contributions.low_controllability,
        ),
        reasons=list(update.reasons),
    )


@router.get("/state/emotion", response_model=EmotionStateResponse)
def emotion_state(
    main_loop: KagyaMainLoop = Depends(get_main_loop),
) -> EmotionStateResponse:
    state = main_loop.emotion_engine.state
    return EmotionStateResponse(
        valence=state.valence,
        arousal=state.arousal,
        optimal_loss=state.optimal_loss,
    )

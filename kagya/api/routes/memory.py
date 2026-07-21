"""Memory routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_memory_system,
    require_admin,
)
from kagya.api.schemas.memory import (
    EpisodeMemoryResponse,
    MemoryMetadataUpdateRequest,
    MemoryReviewRequest,
    MemorySearchResponse,
    SemanticMemoryResponse,
    SemanticGraphResponse,
    SemanticLifecycleRequest,
    SemanticRelationshipRequest,
    SemanticPolicyRequest,
)
from kagya.memory import DualMemorySystem, EpisodicMemoryRecord, SemanticMemoryRecord
from kagya.memory import MemoryLifecycleStatus, ValidationStatus
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_admin)]
)


@router.get("/search", response_model=MemorySearchResponse)
def search_memory(
    query: str,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> MemorySearchResponse:
    context = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_READ,
        source="api.memory.search",
        handler=lambda: memory.retrieve_context(query),
    ).value
    return MemorySearchResponse(
        db1_results=[episode_response(record) for record in context.db1_results],
        db2_results=[semantic_response(record) for record in context.db2_results],
    )


@router.get("/episodes/{episode_id}", response_model=EpisodeMemoryResponse)
def get_episode(
    episode_id: str,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> EpisodeMemoryResponse:
    record = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_READ,
        source="api.memory.episode",
        handler=lambda: memory.get_episodic(episode_id),
    ).value
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(record)


@router.get("/semantic/{memory_id}", response_model=SemanticMemoryResponse)
def get_semantic(
    memory_id: str,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SemanticMemoryResponse:
    record = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_READ,
        source="api.memory.semantic",
        handler=lambda: memory.get_semantic(memory_id),
    ).value
    if record is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return semantic_response(record)


@router.post("/episodes/{episode_id}/archive", response_model=EpisodeMemoryResponse)
def archive_episode(
    episode_id: str,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> EpisodeMemoryResponse:
    record = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_UPDATE,
        source="api.memory.episode.archive",
        handler=lambda: memory.archive_episodic(episode_id),
    ).value
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(record)


@router.post("/episodes/{episode_id}/metadata", response_model=EpisodeMemoryResponse)
def update_episode_metadata(
    episode_id: str,
    request: MemoryMetadataUpdateRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> EpisodeMemoryResponse:
    record = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_UPDATE,
        source="api.memory.episode.metadata",
        handler=lambda: memory.update_episodic_metadata(
            episode_id, tags=request.tags, operator_metadata=request.operator_metadata
        ),
    ).value
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(record)


@router.post("/episodes/{episode_id}/review", response_model=EpisodeMemoryResponse)
def review_episode(
    episode_id: str,
    request: MemoryReviewRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> EpisodeMemoryResponse:
    try:
        validation = ValidationStatus(request.validation_status)
        lifecycle = MemoryLifecycleStatus(request.lifecycle_status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_UPDATE,
        source="api.memory.episode.review",
        handler=lambda: memory.review_episodic(
            episode_id,
            validation_status=validation,
            lifecycle_status=lifecycle,
        ),
    ).value
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(record)


@router.post("/semantic/{memory_id}/archive", response_model=SemanticMemoryResponse)
def archive_semantic(
    memory_id: str,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SemanticMemoryResponse:
    record = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_UPDATE,
        source="api.memory.semantic.archive",
        handler=lambda: memory.archive_semantic(memory_id),
    ).value
    if record is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return semantic_response(record)


@router.post("/semantic/{memory_id}/lifecycle", response_model=SemanticMemoryResponse)
def update_semantic_lifecycle(
    memory_id: str,
    request: SemanticLifecycleRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SemanticMemoryResponse:
    operations = {
        "archive": lambda: memory.archive_semantic(
            memory_id, idempotency_key=request.idempotency_key
        ),
        "restore": lambda: memory.restore_semantic(
            memory_id, idempotency_key=request.idempotency_key
        ),
        "forget": lambda: memory.forget_semantic(
            memory_id, idempotency_key=request.idempotency_key
        ),
    }
    operation = operations.get(request.action)
    if operation is None:
        raise HTTPException(
            status_code=400, detail="Unsupported semantic lifecycle action"
        )
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.MEMORY_UPDATE,
            source=f"api.memory.semantic.{request.action}",
            handler=operation,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return semantic_response(record)


@router.delete("/semantic/{memory_id}", status_code=204)
def delete_semantic(
    memory_id: str,
    idempotency_key: str,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> None:
    deleted = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_UPDATE,
        source="api.memory.semantic.delete",
        handler=lambda: memory.delete_semantic(
            memory_id, idempotency_key=idempotency_key
        ),
    ).value
    if not deleted:
        raise HTTPException(status_code=404, detail="Semantic memory not found")


@router.post(
    "/semantic/{memory_id}/relationships", response_model=SemanticMemoryResponse
)
def relate_semantic(
    memory_id: str,
    request: SemanticRelationshipRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SemanticMemoryResponse:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.MEMORY_UPDATE,
            source="api.memory.semantic.relationship",
            handler=lambda: memory.propose_semantic_relationship(
                memory_id,
                target_id=request.target_id,
                relationship=request.relationship,
                idempotency_key=request.idempotency_key,
            ),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return semantic_response(record)


@router.get("/semantic/{memory_id}/graph", response_model=SemanticGraphResponse)
def get_semantic_graph(
    memory_id: str,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SemanticGraphResponse:
    records = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_READ,
        source="api.memory.semantic.graph",
        handler=lambda: memory.semantic_graph(memory_id),
    ).value
    if not records:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return SemanticGraphResponse(records=[semantic_response(item) for item in records])


@router.post("/semantic/{memory_id}/policy", response_model=SemanticMemoryResponse)
def update_semantic_policy(
    memory_id: str,
    request: SemanticPolicyRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SemanticMemoryResponse:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.MEMORY_UPDATE,
            source="api.memory.semantic.policy",
            handler=lambda: memory.update_semantic_policy(
                memory_id,
                idempotency_key=request.idempotency_key,
                confidence=request.confidence,
                validity=request.validity,
                valid_from=request.valid_from,
                valid_until=request.valid_until,
                expires_at=request.expires_at,
                decay_rate=request.decay_rate,
            ),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return semantic_response(record)


@router.post("/semantic/{memory_id}/metadata", response_model=SemanticMemoryResponse)
def update_semantic_metadata(
    memory_id: str,
    request: MemoryMetadataUpdateRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SemanticMemoryResponse:
    record = execute_agent_event(
        runtime,
        AgentEventType.MEMORY_UPDATE,
        source="api.memory.semantic.metadata",
        handler=lambda: memory.update_semantic_metadata(
            memory_id, tags=request.tags, operator_metadata=request.operator_metadata
        ),
    ).value
    if record is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return semantic_response(record)


def episode_response(record: EpisodicMemoryRecord) -> EpisodeMemoryResponse:
    return EpisodeMemoryResponse(
        id=record.id,
        user_input=record.user_input,
        response=record.response,
        loss=record.loss,
        emotion_valence=record.emotion_valence,
        emotion_arousal=record.emotion_arousal,
        record_type=record.record_type.value,
        archived=record.archived,
        created_at=record.created_at,
        tags=record.tags,
        operator_metadata=record.operator_metadata,
        validation_status=record.validation_status.value,
        lifecycle_status=record.lifecycle_status.value,
        generation_healthy=record.generation_health.healthy,
        generation_health_reasons=record.generation_health.reasons,
        content_hash=record.content_hash,
        source_event_id=record.source_event_id,
        source=record.source,
        processing_sequence=record.processing_sequence,
        provider=record.provider,
        model_id=record.model_id,
        model_revision=record.model_revision,
        adapter_id=record.adapter_id,
        consolidation_status=record.consolidation_status.value,
        context_id=record.context_id,
        source_channel=record.source_channel,
        source_session_id=record.source_session_id,
        semantic_relevance=record.semantic_relevance,
        context_compatibility=record.context_compatibility,
        context_relation=record.context_relation,
        cross_context=record.cross_context,
        experience_id=record.experience_id,
        subjective_salience=record.subjective_salience,
        autobiographical_importance=record.autobiographical_importance,
        supersedes_id=record.supersedes_id,
        corrected_by_id=record.corrected_by_id,
        training_included=record.training_included,
        training_exclusion_refs=record.training_exclusion_refs,
    )


def semantic_response(record: SemanticMemoryRecord) -> SemanticMemoryResponse:
    return SemanticMemoryResponse(
        id=record.id,
        text=record.text,
        source_episode_ids=record.source_episode_ids,
        record_type=record.record_type.value,
        archived=record.archived,
        created_at=record.created_at,
        tags=record.tags,
        operator_metadata=record.operator_metadata,
        context_id=record.context_id,
        source_channel=record.source_channel,
        source_session_id=record.source_session_id,
        semantic_relevance=record.semantic_relevance,
        context_compatibility=record.context_compatibility,
        context_relation=record.context_relation,
        cross_context=record.cross_context,
        schema_version=record.schema_version,
        version=record.version,
        content_hash=record.content_hash,
        confidence=record.confidence,
        effective_confidence=record.effective_confidence,
        validity=record.validity,
        valid_from=record.valid_from,
        valid_until=record.valid_until,
        expires_at=record.expires_at,
        decay_rate=record.decay_rate,
        last_confirmed_at=record.last_confirmed_at,
        lifecycle_status=record.lifecycle_status.value,
        supersedes_id=record.supersedes_id,
        superseded_by_id=record.superseded_by_id,
        corrected_by_id=record.corrected_by_id,
        contradiction_ids=record.contradiction_ids,
        source_feedback_ids=record.source_feedback_ids,
        merge_candidate_ids=record.merge_candidate_ids,
        audit_log=record.audit_log,
    )

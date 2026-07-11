"""Memory routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import get_memory_system, require_admin
from kagya.api.schemas.memory import (
    EpisodeMemoryResponse,
    MemoryMetadataUpdateRequest,
    MemorySearchResponse,
    SemanticMemoryResponse,
)
from kagya.memory import DualMemorySystem, EpisodicMemoryRecord, SemanticMemoryRecord


router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_admin)])


@router.get("/search", response_model=MemorySearchResponse)
def search_memory(query: str, memory: DualMemorySystem = Depends(get_memory_system)) -> MemorySearchResponse:
    context = memory.retrieve_context(query)
    return MemorySearchResponse(
        db1_results=[episode_response(record) for record in context.db1_results],
        db2_results=[semantic_response(record) for record in context.db2_results],
    )


@router.get("/episodes/{episode_id}", response_model=EpisodeMemoryResponse)
def get_episode(episode_id: str, memory: DualMemorySystem = Depends(get_memory_system)) -> EpisodeMemoryResponse:
    record = memory.get_episodic(episode_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(record)


@router.get("/semantic/{memory_id}", response_model=SemanticMemoryResponse)
def get_semantic(memory_id: str, memory: DualMemorySystem = Depends(get_memory_system)) -> SemanticMemoryResponse:
    record = memory.get_semantic(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return semantic_response(record)


@router.post("/episodes/{episode_id}/archive", response_model=EpisodeMemoryResponse)
def archive_episode(
    episode_id: str, memory: DualMemorySystem = Depends(get_memory_system)
) -> EpisodeMemoryResponse:
    record = memory.archive_episodic(episode_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(record)


@router.post("/episodes/{episode_id}/metadata", response_model=EpisodeMemoryResponse)
def update_episode_metadata(
    episode_id: str,
    request: MemoryMetadataUpdateRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
) -> EpisodeMemoryResponse:
    record = memory.update_episodic_metadata(
        episode_id, tags=request.tags, operator_metadata=request.operator_metadata
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(record)


@router.post("/semantic/{memory_id}/archive", response_model=SemanticMemoryResponse)
def archive_semantic(
    memory_id: str, memory: DualMemorySystem = Depends(get_memory_system)
) -> SemanticMemoryResponse:
    record = memory.archive_semantic(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    return semantic_response(record)


@router.post("/semantic/{memory_id}/metadata", response_model=SemanticMemoryResponse)
def update_semantic_metadata(
    memory_id: str,
    request: MemoryMetadataUpdateRequest,
    memory: DualMemorySystem = Depends(get_memory_system),
) -> SemanticMemoryResponse:
    record = memory.update_semantic_metadata(
        memory_id, tags=request.tags, operator_metadata=request.operator_metadata
    )
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
    )

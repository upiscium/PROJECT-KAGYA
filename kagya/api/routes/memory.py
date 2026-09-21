"""Memory routes."""

import json

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import get_memory_system, require_admin
from kagya.api.schemas.memory import (
    EpisodeMemoryResponse,
    MemorySearchResponse,
    SemanticMemoryResponse,
)
from kagya.memory import (
    DualMemorySystem,
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
    EpisodicMemoryRecord,
    SemanticMemoryRecord,
)


router = APIRouter(
    prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_admin)]
)


@router.get("/search", response_model=MemorySearchResponse)
def search_memory(
    query: str, memory: DualMemorySystem = Depends(get_memory_system)
) -> MemorySearchResponse:
    context = memory.retrieve_context(query)
    return MemorySearchResponse(
        db1_results=[episode_response(record) for record in context.db1_results],
        db2_results=[semantic_response(record) for record in context.db2_results],
    )


@router.get("/episodes/{episode_id}", response_model=EpisodeMemoryResponse)
def get_episode(
    episode_id: str, memory: DualMemorySystem = Depends(get_memory_system)
) -> EpisodeMemoryResponse:
    try:
        committed = memory.get_committed_episodic(episode_id)
    except EpisodicMemoryReadError:
        raise HTTPException(
            status_code=503, detail="Committed episodic Memory is unavailable"
        ) from None
    except EpisodicMemoryFormatError:
        raise HTTPException(
            status_code=500, detail="Committed episodic Memory is invalid"
        ) from None
    if committed is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode_response(committed.record)


@router.get("/semantic/{memory_id}", response_model=SemanticMemoryResponse)
def get_semantic(
    memory_id: str, memory: DualMemorySystem = Depends(get_memory_system)
) -> SemanticMemoryResponse:
    result = memory.db2.get(ids=[memory_id], include=["documents", "metadatas"])
    ids = result.get("ids") or []
    if not ids:
        raise HTTPException(status_code=404, detail="Semantic memory not found")
    metadata = (result.get("metadatas") or [{}])[0] or {}
    document = (result.get("documents") or [""])[0] or ""
    source_ids = metadata.get("source_episode_ids", "[]")
    return SemanticMemoryResponse(
        id=memory_id,
        text=str(metadata.get("text", document)),
        source_episode_ids=json.loads(source_ids) if isinstance(source_ids, str) else [],
        record_type=str(metadata.get("record_type", "semantic_memory")),
        created_at=str(metadata.get("created_at", "")),
    )


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
    )


def semantic_response(record: SemanticMemoryRecord) -> SemanticMemoryResponse:
    return SemanticMemoryResponse(
        id=record.id,
        text=record.text,
        source_episode_ids=record.source_episode_ids,
        record_type=record.record_type.value,
        created_at=record.created_at,
    )

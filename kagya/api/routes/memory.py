"""Memory routes."""

import json

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import get_memory_system
from kagya.api.schemas.memory import EpisodeMemoryResponse, MemorySearchResponse, SemanticMemoryResponse
from kagya.memory import DualMemorySystem, EpisodicMemoryRecord, SemanticMemoryRecord


router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/search", response_model=MemorySearchResponse)
def search_memory(query: str, memory: DualMemorySystem = Depends(get_memory_system)) -> MemorySearchResponse:
    context = memory.retrieve_context(query)
    return MemorySearchResponse(
        db1_results=[episode_response(record) for record in context.db1_results],
        db2_results=[semantic_response(record) for record in context.db2_results],
    )


@router.get("/episodes/{episode_id}", response_model=EpisodeMemoryResponse)
def get_episode(episode_id: str, memory: DualMemorySystem = Depends(get_memory_system)) -> EpisodeMemoryResponse:
    result = memory.db1.get(ids=[episode_id], include=["metadatas"])
    ids = result.get("ids") or []
    if not ids:
        raise HTTPException(status_code=404, detail="Episode not found")
    metadata = (result.get("metadatas") or [{}])[0] or {}
    return EpisodeMemoryResponse(
        id=episode_id,
        user_input=str(metadata.get("user_input", "")),
        response=str(metadata.get("response", "")),
        hidden_thought=str(metadata.get("hidden_thought", "")),
        loss=float(metadata.get("loss", 0.0)),
        emotion_valence=float(metadata.get("emotion_valence", 0.0)),
        emotion_arousal=float(metadata.get("emotion_arousal", 0.0)),
        record_type=str(metadata.get("record_type", "episodic_log")),
        archived=bool(metadata.get("archived", False)),
        created_at=str(metadata.get("created_at", "")),
    )


@router.get("/semantic/{memory_id}", response_model=SemanticMemoryResponse)
def get_semantic(memory_id: str, memory: DualMemorySystem = Depends(get_memory_system)) -> SemanticMemoryResponse:
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
        hidden_thought=record.hidden_thought,
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

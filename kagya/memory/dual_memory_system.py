"""ChromaDB-backed dual memory implementation."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.types import Metadata
import numpy as np

from kagya.config import Settings
from kagya.memory.consolidation import build_consolidation_prompt
from kagya.memory.memory_evaluator import MemoryEvaluator
from kagya.memory.memory_schema import (
    EpisodicMemoryRecord,
    MemoryContext,
    MemoryRecordType,
    SemanticMemoryRecord,
)
from kagya.models import ModelProvider


class DeterministicEmbeddingFunction:
    """Small deterministic embedding function for local tests and bootstrap use."""

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return [_embed_text(text) for text in input]

    def embed_query(
        self, input: Sequence[str]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.asarray(self(input), dtype=np.float32)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self(input)

    @staticmethod
    def name() -> str:
        return "default"

    @staticmethod
    def is_legacy() -> bool:
        return True


class DualMemorySystem:
    """Dual memory backed by DB1 hippocampus and DB2 cortex Chroma collections."""

    def __init__(
        self,
        settings: Settings,
        embedding_function: Any | None = None,
        evaluator: MemoryEvaluator | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_function: Any = (
            embedding_function
            if embedding_function is not None
            else DeterministicEmbeddingFunction()
        )
        self.evaluator = evaluator or MemoryEvaluator()
        self.client = chromadb.PersistentClient(path=str(settings.memory.persist_directory))
        self.db1 = self.client.get_or_create_collection(
            name=settings.memory.db1_collection,
            embedding_function=self.embedding_function,
        )
        self.db2 = self.client.get_or_create_collection(
            name=settings.memory.db2_collection,
            embedding_function=self.embedding_function,
        )

    def save_episodic(
        self,
        user_input: str,
        response: str,
        *,
        hidden_thought: str = "",
        loss: float = 0.0,
        emotion_valence: float = 0.0,
        emotion_arousal: float = 0.0,
        record_type: MemoryRecordType = MemoryRecordType.EPISODIC_LOG,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        episode_id = f"episode-{uuid4()}"
        created_at = _now_iso()
        record_metadata: Metadata = {
            "user_input": user_input,
            "response": response,
            "hidden_thought": hidden_thought,
            "loss": float(loss),
            "emotion_valence": float(emotion_valence),
            "emotion_arousal": float(emotion_arousal),
            "record_type": record_type.value,
            "archived": False,
            "created_at": created_at,
            "extra": json.dumps(metadata or {}),
        }
        self.db1.add(
            ids=[episode_id],
            documents=[_episodic_document(user_input, response, hidden_thought)],
            metadatas=[record_metadata],
        )
        return episode_id

    def save_semantic(
        self,
        text: str,
        *,
        source_episode_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        semantic_id = f"semantic-{uuid4()}"
        record_metadata: Metadata = {
            "text": text,
            "source_episode_ids": json.dumps(source_episode_ids or []),
            "record_type": MemoryRecordType.SEMANTIC_MEMORY.value,
            "created_at": _now_iso(),
            "extra": json.dumps(metadata or {}),
        }
        self.db2.add(ids=[semantic_id], documents=[text], metadatas=[record_metadata])
        return semantic_id

    def retrieve_context(self, query: str) -> MemoryContext:
        db1_results = self.db1.query(
            query_texts=[query],
            n_results=self.settings.memory.db1_top_k,
            where={"archived": False},
        )
        db2_results = self.db2.query(
            query_texts=[query],
            n_results=self.settings.memory.db2_top_k,
        )
        return MemoryContext(
            db1_results=_episodic_records_from_query(db1_results),
            db2_results=_semantic_records_from_query(db2_results),
        )

    def consolidate_to_semantic(self, model_provider: ModelProvider) -> list[str]:
        records = self._get_unarchived_episodic_records()
        semantic_ids: list[str] = []
        for record in records:
            if not self.evaluator.should_consolidate(record):
                continue
            semantic_text = model_provider.generate(build_consolidation_prompt(record))
            semantic_ids.append(
                self.save_semantic(semantic_text, source_episode_ids=[record.id])
            )
            self._archive_episodic(record.id)
        return semantic_ids

    def _get_unarchived_episodic_records(self) -> list[EpisodicMemoryRecord]:
        result = self.db1.get(where={"archived": False})
        return _episodic_records_from_get(result)

    def _archive_episodic(self, episode_id: str) -> None:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return
        metadata = dict(metadatas[0])
        metadata["archived"] = True
        self.db1.update(ids=[episode_id], metadatas=[metadata])


def _embed_text(text: str) -> list[float]:
    buckets = [0.0] * 16
    for index, char in enumerate(text):
        buckets[index % len(buckets)] += float(ord(char) % 31) / 31.0
    magnitude = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [value / magnitude for value in buckets]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _episodic_document(user_input: str, response: str, hidden_thought: str) -> str:
    return f"User: {user_input}\nAssistant: {response}\nThought: {hidden_thought}".strip()


def _episodic_records_from_query(
    result: Mapping[str, Any],
) -> list[EpisodicMemoryRecord]:
    ids = _first_result_list(result.get("ids"))
    metadatas = _first_result_list(result.get("metadatas"))
    return [
        _episodic_record_from_metadata(record_id, metadata or {})
        for record_id, metadata in zip(ids, metadatas, strict=False)
    ]


def _semantic_records_from_query(
    result: Mapping[str, Any],
) -> list[SemanticMemoryRecord]:
    ids = _first_result_list(result.get("ids"))
    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))
    return [
        _semantic_record_from_metadata(record_id, document or "", metadata or {})
        for record_id, document, metadata in zip(
            ids, documents, metadatas, strict=False
        )
    ]


def _episodic_records_from_get(
    result: Mapping[str, Any],
) -> list[EpisodicMemoryRecord]:
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    return [
        _episodic_record_from_metadata(record_id, metadata or {})
        for record_id, metadata in zip(ids, metadatas, strict=False)
    ]


def _episodic_record_from_metadata(
    record_id: str, metadata: dict[str, Any]
) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        id=record_id,
        user_input=str(metadata.get("user_input", "")),
        response=str(metadata.get("response", "")),
        hidden_thought=str(metadata.get("hidden_thought", "")),
        loss=float(metadata.get("loss", 0.0)),
        emotion_valence=float(metadata.get("emotion_valence", 0.0)),
        emotion_arousal=float(metadata.get("emotion_arousal", 0.0)),
        record_type=MemoryRecordType(
            str(metadata.get("record_type", MemoryRecordType.EPISODIC_LOG.value))
        ),
        archived=bool(metadata.get("archived", False)),
        created_at=str(metadata.get("created_at", "")),
        metadata=_loads_json_dict(metadata.get("extra")),
    )


def _semantic_record_from_metadata(
    record_id: str, document: str, metadata: dict[str, Any]
) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        id=record_id,
        text=str(metadata.get("text", document)),
        source_episode_ids=_loads_json_list(metadata.get("source_episode_ids")),
        record_type=MemoryRecordType(
            str(metadata.get("record_type", MemoryRecordType.SEMANTIC_MEMORY.value))
        ),
        created_at=str(metadata.get("created_at", "")),
        metadata=_loads_json_dict(metadata.get("extra")),
    )


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    return value[0] if isinstance(value[0], list) else value


def _loads_json_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _loads_json_list(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in loaded] if isinstance(loaded, list) else []

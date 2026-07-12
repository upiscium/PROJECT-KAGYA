"""ChromaDB-backed dual memory implementation."""

from collections.abc import Sequence
from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Any
from uuid import uuid4

import chromadb

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

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self(input)

    @staticmethod
    def name() -> str:
        return "default"

    @staticmethod
    def is_legacy() -> bool:
        return True


class SentenceTransformerEmbeddingFunction:
    """Chroma embedding function backed by a configured sentence-transformers model."""

    def __init__(self, model_id: str, model_loader: Any | None = None) -> None:
        self.model_id = model_id
        self._model_loader = model_loader or _load_sentence_transformer
        self._model: Any | None = None

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self._encode(input)

    def name(self) -> str:
        return f"sentence-transformers:{self.model_id}"

    @staticmethod
    def is_legacy() -> bool:
        return False

    def _encode(self, input: Sequence[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(list(input), normalize_embeddings=True)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(vector) for vector in embeddings]

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(self.model_id)
        return self._model


def create_embedding_function(settings: Settings) -> Any:
    """Create the configured memory embedding function."""

    model_id = settings.memory.embedding_model_id
    if model_id == "deterministic":
        return DeterministicEmbeddingFunction()
    return SentenceTransformerEmbeddingFunction(model_id)


class DualMemorySystem:
    """Dual memory backed by DB1 hippocampus and DB2 cortex Chroma collections."""

    def __init__(
        self,
        settings: Settings,
        embedding_function: Any | None = None,
        evaluator: MemoryEvaluator | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_function = embedding_function or create_embedding_function(settings)
        self.evaluator = evaluator or MemoryEvaluator()
        self.client = chromadb.PersistentClient(path=str(settings.memory.persist_directory))
        db1_collection = _collection_name_for_embedding(
            settings.memory.db1_collection, self.embedding_function
        )
        db2_collection = _collection_name_for_embedding(
            settings.memory.db2_collection, self.embedding_function
        )
        self.db1 = self.client.get_or_create_collection(
            name=db1_collection,
            embedding_function=self.embedding_function,
            metadata={"kagya_embedding": _embedding_name(self.embedding_function)},
        )
        self.db2 = self.client.get_or_create_collection(
            name=db2_collection,
            embedding_function=self.embedding_function,
            metadata={"kagya_embedding": _embedding_name(self.embedding_function)},
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
        record_metadata = {
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
        record_metadata = {
            "text": text,
            "source_episode_ids": json.dumps(source_episode_ids or []),
            "record_type": MemoryRecordType.SEMANTIC_MEMORY.value,
            "archived": False,
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
            db2_results=[
                record
                for record in _semantic_records_from_query(db2_results)
                if not record.archived
            ],
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
        self.archive_episodic(episode_id)

    def get_episodic(self, episode_id: str) -> EpisodicMemoryRecord | None:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        return _first_episode_from_get(result)

    def get_semantic(self, memory_id: str) -> SemanticMemoryRecord | None:
        result = self.db2.get(ids=[memory_id], include=["documents", "metadatas"])
        return _first_semantic_from_get(result)

    def archive_episodic(self, episode_id: str) -> EpisodicMemoryRecord | None:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        metadata["archived"] = True
        self.db1.update(ids=[episode_id], metadatas=[metadata])
        return self.get_episodic(episode_id)

    def archive_semantic(self, memory_id: str) -> SemanticMemoryRecord | None:
        result = self.db2.get(ids=[memory_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        metadata["archived"] = True
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        return self.get_semantic(memory_id)

    def update_episodic_metadata(
        self,
        episode_id: str,
        *,
        tags: list[str] | None = None,
        operator_metadata: dict[str, Any] | None = None,
    ) -> EpisodicMemoryRecord | None:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        metadata["extra"] = json.dumps(
            _updated_operator_extra(metadata.get("extra"), tags, operator_metadata)
        )
        self.db1.update(ids=[episode_id], metadatas=[metadata])
        return self.get_episodic(episode_id)

    def update_semantic_metadata(
        self,
        memory_id: str,
        *,
        tags: list[str] | None = None,
        operator_metadata: dict[str, Any] | None = None,
    ) -> SemanticMemoryRecord | None:
        result = self.db2.get(ids=[memory_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        metadata["extra"] = json.dumps(
            _updated_operator_extra(metadata.get("extra"), tags, operator_metadata)
        )
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        return self.get_semantic(memory_id)


def _embed_text(text: str) -> list[float]:
    buckets = [0.0] * 16
    for index, char in enumerate(text):
        buckets[index % len(buckets)] += float(ord(char) % 31) / 31.0
    magnitude = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [value / magnitude for value in buckets]


def _collection_name_for_embedding(base_name: str, embedding_function: Any) -> str:
    if _is_legacy_embedding(embedding_function):
        return base_name
    embedding_name = _embedding_name(embedding_function)
    digest = hashlib.sha256(embedding_name.encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", embedding_name).strip("-")[:32]
    return f"{base_name}-{safe_name}-{digest}"


def _embedding_name(embedding_function: Any) -> str:
    name = getattr(embedding_function, "name", None)
    if callable(name):
        return str(name())
    return embedding_function.__class__.__name__


def _is_legacy_embedding(embedding_function: Any) -> bool:
    is_legacy = getattr(embedding_function, "is_legacy", None)
    return bool(callable(is_legacy) and is_legacy())


def _load_sentence_transformer(model_id: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for configured memory embeddings") from exc
    return SentenceTransformer(model_id)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _episodic_document(user_input: str, response: str, hidden_thought: str) -> str:
    return f"User: {user_input}\nAssistant: {response}\nThought: {hidden_thought}".strip()


def _episodic_records_from_query(result: dict[str, Any]) -> list[EpisodicMemoryRecord]:
    ids = _first_result_list(result.get("ids"))
    metadatas = _first_result_list(result.get("metadatas"))
    return [_episodic_record_from_metadata(record_id, metadata or {}) for record_id, metadata in zip(ids, metadatas, strict=False)]


def _semantic_records_from_query(result: dict[str, Any]) -> list[SemanticMemoryRecord]:
    ids = _first_result_list(result.get("ids"))
    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))
    return [
        _semantic_record_from_metadata(record_id, document or "", metadata or {})
        for record_id, document, metadata in zip(ids, documents, metadatas, strict=False)
    ]


def _episodic_records_from_get(result: dict[str, Any]) -> list[EpisodicMemoryRecord]:
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    return [_episodic_record_from_metadata(record_id, metadata or {}) for record_id, metadata in zip(ids, metadatas, strict=False)]


def _semantic_records_from_get(result: dict[str, Any]) -> list[SemanticMemoryRecord]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        _semantic_record_from_metadata(record_id, document or "", metadata or {})
        for record_id, document, metadata in zip(ids, documents, metadatas, strict=False)
    ]


def _first_episode_from_get(result: dict[str, Any]) -> EpisodicMemoryRecord | None:
    records = _episodic_records_from_get(result)
    return records[0] if records else None


def _first_semantic_from_get(result: dict[str, Any]) -> SemanticMemoryRecord | None:
    records = _semantic_records_from_get(result)
    return records[0] if records else None


def _episodic_record_from_metadata(record_id: str, metadata: dict[str, Any]) -> EpisodicMemoryRecord:
    extra = _loads_json_dict(metadata.get("extra"))
    return EpisodicMemoryRecord(
        id=record_id,
        user_input=str(metadata.get("user_input", "")),
        response=str(metadata.get("response", "")),
        hidden_thought=str(metadata.get("hidden_thought", "")),
        loss=float(metadata.get("loss", 0.0)),
        emotion_valence=float(metadata.get("emotion_valence", 0.0)),
        emotion_arousal=float(metadata.get("emotion_arousal", 0.0)),
        record_type=MemoryRecordType(str(metadata.get("record_type", MemoryRecordType.EPISODIC_LOG.value))),
        archived=bool(metadata.get("archived", False)),
        created_at=str(metadata.get("created_at", "")),
        metadata=extra,
        tags=_operator_tags(extra),
        operator_metadata=_operator_metadata(extra),
    )


def _semantic_record_from_metadata(record_id: str, document: str, metadata: dict[str, Any]) -> SemanticMemoryRecord:
    extra = _loads_json_dict(metadata.get("extra"))
    return SemanticMemoryRecord(
        id=record_id,
        text=str(metadata.get("text", document)),
        source_episode_ids=_loads_json_list(metadata.get("source_episode_ids")),
        record_type=MemoryRecordType(str(metadata.get("record_type", MemoryRecordType.SEMANTIC_MEMORY.value))),
        archived=bool(metadata.get("archived", False)),
        created_at=str(metadata.get("created_at", "")),
        metadata=extra,
        tags=_operator_tags(extra),
        operator_metadata=_operator_metadata(extra),
    )


def _first_metadata(result: dict[str, Any]) -> dict[str, Any] | None:
    metadatas = result.get("metadatas") or []
    if not metadatas:
        return None
    return dict(metadatas[0] or {})


def _updated_operator_extra(
    raw_extra: Any,
    tags: list[str] | None,
    operator_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    extra = _loads_json_dict(raw_extra)
    if tags is not None:
        extra["tags"] = _clean_tags(tags)
    if operator_metadata is not None:
        extra["operator_metadata"] = operator_metadata
    return extra


def _clean_tags(tags: list[str]) -> list[str]:
    cleaned: list[str] = []
    for tag in tags:
        normalized = tag.strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _operator_tags(extra: dict[str, Any]) -> list[str]:
    tags = extra.get("tags")
    if not isinstance(tags, list):
        return []
    return [tag for tag in tags if isinstance(tag, str)]


def _operator_metadata(extra: dict[str, Any]) -> dict[str, Any]:
    metadata = extra.get("operator_metadata")
    return metadata if isinstance(metadata, dict) else {}


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

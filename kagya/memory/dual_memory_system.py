"""ChromaDB-backed dual memory implementation."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.types import Metadata

from kagya.config import Settings
from kagya.memory.consolidation import build_consolidation_prompt
from kagya.memory.memory_evaluator import MemoryEvaluator
from kagya.memory.memory_schema import (
    EpisodicMemoryRecord,
    ConsolidationStatus,
    GenerationHealth,
    MemoryLifecycleStatus,
    MemoryRecordKind,
    MemoryContext,
    MemoryRecordType,
    SemanticMemoryRecord,
    ValidationStatus,
)
from kagya.memory.quality import assess_generation_health
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
        self.embedding_function = embedding_function or create_embedding_function(
            settings
        )
        self.evaluator = evaluator or MemoryEvaluator()
        self.client = chromadb.PersistentClient(
            path=str(settings.memory.persist_directory)
        )
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
        generation_health: GenerationHealth | None = None,
        source_event_id: str | None = None,
        source: str = "unknown",
        source_channel: str = "unknown",
        source_session_id: str | None = None,
        processing_sequence: int | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
        context_id: str | None = None,
        provider: str = "unknown",
        model_id: str = "unknown",
        model_revision: str = "unknown",
        adapter_id: str | None = None,
        validation_status: ValidationStatus = ValidationStatus.VERIFIED,
    ) -> str:
        health = generation_health or GenerationHealth()
        lifecycle = (
            MemoryLifecycleStatus.ACTIVE
            if health.healthy
            else MemoryLifecycleStatus.QUARANTINED
        )
        content_hash = _content_hash(user_input, response)
        dedup_key = _dedup_key(source_event_id, content_hash)
        existing = self.db1.get(where={"dedup_key": dedup_key})
        if existing.get("ids"):
            return str(existing["ids"][0])
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
            "schema_version": 3,
            "lifecycle_status": lifecycle.value,
            "validation_status": validation_status.value,
            "content_hash": content_hash,
            "dedup_key": dedup_key,
            "consolidation_status": ConsolidationStatus.PENDING.value,
            "experience_id": "",
            "subjective_salience": 0.0,
            "autobiographical_importance": 0.0,
            "training_included": True,
            "training_exclusion_refs": "[]",
            "extra": json.dumps(
                {
                    **(metadata or {}),
                    "generation_health": health.__dict__,
                    "provenance": {
                        "source_event_id": source_event_id,
                        "source": source,
                        "source_channel": source_channel,
                        "source_session_id": source_session_id,
                        "processing_sequence": processing_sequence,
                        "causation_id": causation_id,
                        "correlation_id": correlation_id,
                        "context_id": context_id,
                        "provider": provider,
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "adapter_id": adapter_id,
                    },
                }
            ),
        }
        self.db1.add(
            ids=[episode_id],
            documents=[_episodic_document(user_input, response, hidden_thought)],
            metadatas=[record_metadata],
        )
        return episode_id

    def link_experience(
        self,
        episode_id: str,
        *,
        experience_id: str,
        subjective_salience: float,
        autobiographical_importance: float,
    ) -> EpisodicMemoryRecord:
        for name, value in (
            ("subjective salience", subjective_salience),
            ("autobiographical importance", autobiographical_importance),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            raise ValueError(f"Unknown episodic memory: {episode_id}")
        metadata["experience_id"] = experience_id
        metadata["subjective_salience"] = subjective_salience
        metadata["autobiographical_importance"] = autobiographical_importance
        self.db1.update(ids=[episode_id], metadatas=[metadata])
        linked = self.get_episodic(episode_id)
        if linked is None:
            raise ValueError(f"Unknown episodic memory: {episode_id}")
        return linked

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
            "archived": False,
            "created_at": _now_iso(),
            "extra": json.dumps(metadata or {}),
        }
        self.db2.add(ids=[semantic_id], documents=[text], metadatas=[record_metadata])
        return semantic_id

    def retrieve_context(
        self,
        query: str,
        *,
        current_context_id: str | None = None,
        context_compatibility: Callable[[str | None], tuple[float, str]] | None = None,
    ) -> MemoryContext:
        db1_results = self.db1.query(
            query_texts=[query],
            n_results=self.settings.memory.db1_top_k * 3,
            where={"archived": False},
        )
        db2_results = self.db2.query(
            query_texts=[query],
            n_results=self.settings.memory.db2_top_k * 3,
        )
        compatibility = context_compatibility or (
            lambda source_id: _default_context_compatibility(
                source_id, current_context_id
            )
        )
        episodic = _annotate_retrieval(
            _episodic_records_from_query(db1_results),
            _first_result_list(db1_results.get("distances")),
            compatibility,
        )
        semantic = _annotate_retrieval(
            _semantic_records_from_query(db2_results),
            _first_result_list(db2_results.get("distances")),
            compatibility,
        )
        return MemoryContext(
            db1_results=[
                record
                for record in episodic
                if record.lifecycle_status == MemoryLifecycleStatus.ACTIVE
            ][: self.settings.memory.db1_top_k],
            db2_results=[
                record
                for record in semantic
                if not record.archived
                and record.metadata.get("publication_status", "published")
                == "published"
            ][: self.settings.memory.db2_top_k],
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
        return [
            record
            for record in _episodic_records_from_get(result)
            if record.lifecycle_status == MemoryLifecycleStatus.ACTIVE
        ]

    def review_episodic(
        self,
        episode_id: str,
        *,
        validation_status: ValidationStatus,
        lifecycle_status: MemoryLifecycleStatus,
    ) -> EpisodicMemoryRecord | None:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        metadata["validation_status"] = validation_status.value
        metadata["lifecycle_status"] = lifecycle_status.value
        self.db1.update(ids=[episode_id], metadatas=[metadata])
        return self.get_episodic(episode_id)

    def apply_feedback_policy(
        self,
        episode_id: str,
        *,
        validation_status: ValidationStatus,
        lifecycle_status: MemoryLifecycleStatus,
        training_included: bool,
        feedback_id: str,
    ) -> EpisodicMemoryRecord:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            raise ValueError(f"Unknown episodic memory: {episode_id}")
        refs = _loads_json_list(metadata.get("training_exclusion_refs"))
        if training_included:
            refs = [item for item in refs if item != feedback_id]
        elif feedback_id not in refs:
            refs.append(feedback_id)
        metadata["validation_status"] = validation_status.value
        metadata["lifecycle_status"] = lifecycle_status.value
        metadata["training_included"] = not refs
        metadata["training_exclusion_refs"] = json.dumps(refs)
        self.db1.update(ids=[episode_id], metadatas=[metadata])
        record = self.get_episodic(episode_id)
        if record is None:
            raise ValueError(f"Unknown episodic memory: {episode_id}")
        return record

    def save_feedback_correction(
        self,
        episode_id: str,
        text: str,
        *,
        feedback_id: str,
        kind: str,
    ) -> str:
        original = self.get_episodic(episode_id)
        if original is None:
            raise ValueError(f"Unknown episodic memory: {episode_id}")
        correction_id = self.save_episodic(
            original.user_input,
            text,
            source_event_id=f"{feedback_id}:{kind}",
            source="explicit_feedback",
            source_channel=original.source_channel,
            source_session_id=original.source_session_id,
            context_id=original.context_id,
            provider="human_feedback",
            model_id="not_applicable",
            model_revision="not_applicable",
            validation_status=ValidationStatus.VERIFIED,
            metadata={
                "feedback_id": feedback_id,
                "feedback_content_kind": kind,
                "supersedes_id": episode_id,
            },
        )
        original_result = self.db1.get(ids=[episode_id], include=["metadatas"])
        original_metadata = _first_metadata(original_result)
        correction_result = self.db1.get(ids=[correction_id], include=["metadatas"])
        correction_metadata = _first_metadata(correction_result)
        if original_metadata is None or correction_metadata is None:
            raise ValueError("Feedback correction persistence failed")
        original_metadata["corrected_by_id"] = correction_id
        original_metadata["lifecycle_status"] = MemoryLifecycleStatus.CORRECTED.value
        original_metadata["validation_status"] = ValidationStatus.DISPUTED.value
        correction_metadata["supersedes_id"] = episode_id
        self.db1.update(ids=[episode_id], metadatas=[original_metadata])
        self.db1.update(ids=[correction_id], metadatas=[correction_metadata])
        return correction_id

    def withdraw_feedback_correction(self, episode_id: str, correction_id: str) -> None:
        correction = self.db1.get(ids=[correction_id], include=["metadatas"])
        correction_metadata = _first_metadata(correction)
        if correction_metadata is not None:
            correction_metadata["archived"] = True
            correction_metadata["lifecycle_status"] = (
                MemoryLifecycleStatus.SUPERSEDED.value
            )
            correction_metadata["training_included"] = False
            self.db1.update(ids=[correction_id], metadatas=[correction_metadata])
        original = self.db1.get(ids=[episode_id], include=["metadatas"])
        original_metadata = _first_metadata(original)
        if (
            original_metadata is not None
            and original_metadata.get("corrected_by_id") == correction_id
        ):
            original_metadata["corrected_by_id"] = ""
            self.db1.update(ids=[episode_id], metadatas=[original_metadata])

    def set_consolidation_state(
        self,
        episode_id: str,
        *,
        status: ConsolidationStatus,
        pipeline_version: str,
        attempt_id: str,
    ) -> EpisodicMemoryRecord | None:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        metadata["consolidation_status"] = status.value
        metadata["consolidation_version"] = pipeline_version
        metadata["consolidation_attempt_id"] = attempt_id
        self.db1.update(ids=[episode_id], metadatas=[metadata])
        return self.get_episodic(episode_id)

    def publish_semantic(self, memory_id: str) -> None:
        result = self.db2.get(ids=[memory_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            raise ValueError(f"Unknown semantic memory: {memory_id}")
        extra = _loads_json_dict(metadata.get("extra"))
        extra["publication_status"] = "published"
        metadata["extra"] = json.dumps(extra)
        self.db2.update(ids=[memory_id], metadatas=[metadata])

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
        raise RuntimeError(
            "sentence-transformers is required for configured memory embeddings"
        ) from exc
    return SentenceTransformer(model_id)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _episodic_document(user_input: str, response: str, hidden_thought: str) -> str:
    return (
        f"User: {user_input}\nAssistant: {response}\nThought: {hidden_thought}".strip()
    )


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


def _episodic_records_from_get(result: Mapping[str, Any]) -> list[EpisodicMemoryRecord]:
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    return [
        _episodic_record_from_metadata(record_id, metadata or {})
        for record_id, metadata in zip(ids, metadatas, strict=False)
    ]


def _semantic_records_from_get(result: Mapping[str, Any]) -> list[SemanticMemoryRecord]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        _semantic_record_from_metadata(record_id, document or "", metadata or {})
        for record_id, document, metadata in zip(
            ids, documents, metadatas, strict=False
        )
    ]


def _first_episode_from_get(result: Mapping[str, Any]) -> EpisodicMemoryRecord | None:
    records = _episodic_records_from_get(result)
    return records[0] if records else None


def _first_semantic_from_get(result: Mapping[str, Any]) -> SemanticMemoryRecord | None:
    records = _semantic_records_from_get(result)
    return records[0] if records else None


def _episodic_record_from_metadata(
    record_id: str, metadata: dict[str, Any]
) -> EpisodicMemoryRecord:
    extra = _loads_json_dict(metadata.get("extra"))
    raw_provenance = extra.get("provenance")
    provenance: dict[str, Any] = (
        raw_provenance if isinstance(raw_provenance, dict) else {}
    )
    raw_health_data = extra.get("generation_health")
    health_data: dict[str, Any] = (
        raw_health_data if isinstance(raw_health_data, dict) else {}
    )
    response = str(metadata.get("response", ""))
    loss = float(metadata.get("loss", 0.0))
    health = (
        GenerationHealth(**health_data)
        if health_data
        else assess_generation_health(response, loss=loss, fallback_used=False)
    )
    default_lifecycle = (
        MemoryLifecycleStatus.ACTIVE
        if health.healthy
        else MemoryLifecycleStatus.QUARANTINED
    )
    return EpisodicMemoryRecord(
        id=record_id,
        user_input=str(metadata.get("user_input", "")),
        response=response,
        hidden_thought=str(metadata.get("hidden_thought", "")),
        loss=loss,
        emotion_valence=float(metadata.get("emotion_valence", 0.0)),
        emotion_arousal=float(metadata.get("emotion_arousal", 0.0)),
        record_type=MemoryRecordType(
            str(metadata.get("record_type", MemoryRecordType.EPISODIC_LOG.value))
        ),
        archived=bool(metadata.get("archived", False)),
        created_at=str(metadata.get("created_at", "")),
        metadata=extra,
        tags=_operator_tags(extra),
        operator_metadata=_operator_metadata(extra),
        schema_version=int(metadata.get("schema_version", 1)),
        input_kind=MemoryRecordKind.EXTERNAL_CLAIM,
        response_kind=MemoryRecordKind.GENERATED_RESPONSE,
        validation_status=ValidationStatus(
            str(metadata.get("validation_status", ValidationStatus.UNVERIFIED.value))
        ),
        lifecycle_status=MemoryLifecycleStatus(
            str(metadata.get("lifecycle_status", default_lifecycle.value))
        ),
        generation_health=health,
        content_hash=str(
            metadata.get(
                "content_hash",
                _content_hash(
                    str(metadata.get("user_input", "")),
                    str(metadata.get("response", "")),
                ),
            )
        ),
        dedup_key=str(metadata.get("dedup_key", "")),
        source_event_id=_optional_str(provenance.get("source_event_id")),
        source=str(provenance.get("source", "unknown")),
        source_channel=str(provenance.get("source_channel", "unknown")),
        source_session_id=_optional_str(provenance.get("source_session_id")),
        processing_sequence=_optional_int(provenance.get("processing_sequence")),
        causation_id=_optional_str(provenance.get("causation_id")),
        correlation_id=_optional_str(provenance.get("correlation_id")),
        context_id=_optional_str(provenance.get("context_id")),
        provider=str(provenance.get("provider", "unknown")),
        model_id=str(provenance.get("model_id", "unknown")),
        model_revision=str(provenance.get("model_revision", "unknown")),
        adapter_id=_optional_str(provenance.get("adapter_id")),
        consolidation_status=ConsolidationStatus(
            str(metadata.get("consolidation_status", ConsolidationStatus.PENDING.value))
        ),
        consolidation_version=str(metadata.get("consolidation_version", "")),
        consolidation_attempt_id=_optional_str(
            metadata.get("consolidation_attempt_id")
        ),
        experience_id=_optional_str(metadata.get("experience_id")),
        subjective_salience=float(metadata.get("subjective_salience", 0.0)),
        autobiographical_importance=float(
            metadata.get("autobiographical_importance", 0.0)
        ),
        contradiction_ids=_loads_json_list(metadata.get("contradiction_ids")),
        supersedes_id=_optional_str(metadata.get("supersedes_id")),
        corrected_by_id=_optional_str(metadata.get("corrected_by_id")),
        training_included=bool(metadata.get("training_included", True)),
        training_exclusion_refs=_loads_json_list(
            metadata.get("training_exclusion_refs")
        ),
    )


def _semantic_record_from_metadata(
    record_id: str, document: str, metadata: dict[str, Any]
) -> SemanticMemoryRecord:
    extra = _loads_json_dict(metadata.get("extra"))
    return SemanticMemoryRecord(
        id=record_id,
        text=str(metadata.get("text", document)),
        source_episode_ids=_loads_json_list(metadata.get("source_episode_ids")),
        record_type=MemoryRecordType(
            str(metadata.get("record_type", MemoryRecordType.SEMANTIC_MEMORY.value))
        ),
        archived=bool(metadata.get("archived", False)),
        created_at=str(metadata.get("created_at", "")),
        metadata=extra,
        tags=_operator_tags(extra),
        operator_metadata=_operator_metadata(extra),
        context_id=_optional_str(extra.get("context_id")),
        source=str(extra.get("source", "unknown")),
        source_channel=str(extra.get("source_channel", "unknown")),
        source_session_id=_optional_str(extra.get("source_session_id")),
    )


def _first_metadata(result: Mapping[str, Any]) -> dict[str, Any] | None:
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


def _content_hash(user_input: str, response: str) -> str:
    normalized = json.dumps([user_input.strip(), response.strip()], ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _dedup_key(source_event_id: str | None, content_hash: str) -> str:
    return f"{source_event_id or 'content'}:{content_hash}"


def _optional_str(value: Any) -> str | None:
    return None if value in {None, ""} else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _default_context_compatibility(
    source_context_id: str | None, current_context_id: str | None
) -> tuple[float, str]:
    if source_context_id is None:
        return 0.45, "legacy_unknown"
    if source_context_id == current_context_id:
        return 1.0, "same_context"
    return 0.2, "cross_context"


def _annotate_retrieval(
    records: list[Any],
    distances: list[Any],
    compatibility: Callable[[str | None], tuple[float, str]],
) -> list[Any]:
    annotated: list[Any] = []
    for index, record in enumerate(records):
        distance = float(distances[index]) if index < len(distances) else 1.0
        relevance = 1.0 / (1.0 + max(0.0, distance))
        context_score, relation = compatibility(record.context_id)
        annotated.append(
            replace(
                record,
                semantic_relevance=relevance,
                context_compatibility=context_score,
                context_relation=relation,
                cross_context=record.context_id is not None
                and relation != "same_context",
            )
        )
    return sorted(
        annotated,
        key=lambda item: (
            0.6 * item.semantic_relevance
            + 0.25 * item.context_compatibility
            + 0.15 * getattr(item, "subjective_salience", 0.0)
        ),
        reverse=True,
    )

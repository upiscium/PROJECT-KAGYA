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
    SemanticLifecycleStatus,
    ValidationStatus,
)
from kagya.memory.quality import assess_generation_health
from kagya.models import ModelProvider
from kagya.external_transaction import (
    ExternalTransactionAudit,
    ExternalTransactionRecord,
    ExternalTransactionStatus,
)


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
        self._external_failure_injector: Callable[[str, str], None] | None = None
        self._scrub_legacy_hidden_thoughts()
        self._backfill_episodic_transactions()
        self._backfill_semantic_records()

    def save_episodic(
        self,
        user_input: str,
        response: str,
        *,
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
        stage_external: bool = False,
    ) -> str:
        _reject_private_transaction_metadata(metadata or {})
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
        transaction_status = (
            ExternalTransactionStatus.PENDING
            if stage_external
            else ExternalTransactionStatus.COMMITTED
        )
        if stage_external and (source_event_id is None or processing_sequence is None):
            raise ValueError("staged external record requires event ID and sequence")
        transaction_id = f"external-{episode_id}"
        transaction_audit = [_external_audit_entry(1, transaction_status, "prepare")]
        record_metadata: Metadata = {
            "user_input": user_input,
            "response": response,
            "loss": float(loss),
            "emotion_valence": float(emotion_valence),
            "emotion_arousal": float(emotion_arousal),
            "record_type": record_type.value,
            "archived": False,
            "created_at": created_at,
            "schema_version": 3,
            "source_event_id": source_event_id or "",
            "processing_sequence": processing_sequence
            if processing_sequence is not None
            else -1,
            "external_transaction_id": transaction_id,
            "external_transaction_schema_version": 1,
            "external_transaction_revision": 1,
            "external_transaction_status": transaction_status.value,
            "external_transaction_updated_at": created_at,
            "external_transaction_audit": json.dumps(transaction_audit),
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
            documents=[_episodic_document(user_input, response)],
            metadatas=[record_metadata],
        )
        return episode_id

    def _scrub_legacy_hidden_thoughts(self) -> None:
        result = self.db1.get(include=["documents", "metadatas"])
        for record_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            metadata = dict(raw_metadata or {})
            if "hidden_thought" not in metadata:
                continue
            metadata.pop("hidden_thought", None)
            document = _episodic_document(
                str(metadata.get("user_input", "")),
                str(metadata.get("response", "")),
            )
            self.db1.delete(ids=[str(record_id)])
            self.db1.add(
                ids=[str(record_id)],
                documents=[document],
                metadatas=[metadata],
            )

    def set_external_failure_injector(
        self, injector: Callable[[str, str], None] | None
    ) -> None:
        """Install a deterministic boundary failure hook for verification."""

        self._external_failure_injector = injector

    def list_external_transactions(self) -> list[ExternalTransactionRecord]:
        result = self.db1.get(include=["metadatas"])
        records: list[ExternalTransactionRecord] = []
        for episode_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            metadata = dict(raw_metadata or {})
            extra = _loads_json_dict(metadata.get("extra"))
            provenance = extra.get("provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            status = ExternalTransactionStatus(
                str(
                    metadata.get(
                        "external_transaction_status",
                        ExternalTransactionStatus.COMMITTED.value,
                    )
                )
            )
            created_at = str(metadata.get("created_at", ""))
            audit_values = _loads_json_dict_list(
                metadata.get("external_transaction_audit")
            )
            records.append(
                ExternalTransactionRecord(
                    schema_version=_metadata_int(
                        metadata.get("external_transaction_schema_version"), 1
                    ),
                    revision=_metadata_int(
                        metadata.get("external_transaction_revision"), 1
                    ),
                    transaction_id=str(
                        metadata.get(
                            "external_transaction_id", f"external-{episode_id}"
                        )
                    ),
                    artifact_type="episodic_chroma",
                    artifact_id=str(episode_id),
                    status=status,
                    event_id=_optional_str(
                        metadata.get("source_event_id")
                        or provenance.get("source_event_id")
                    ),
                    processing_sequence=_external_sequence(
                        metadata.get("processing_sequence"),
                        provenance.get("processing_sequence"),
                    ),
                    source=str(provenance.get("source", "unknown")),
                    causation_id=_optional_str(provenance.get("causation_id")),
                    correlation_id=_optional_str(provenance.get("correlation_id")),
                    created_at=created_at,
                    updated_at=str(
                        metadata.get("external_transaction_updated_at", created_at)
                    ),
                    audit=[
                        ExternalTransactionAudit.model_validate(item)
                        for item in audit_values
                    ],
                )
            )
        return records

    def finalize_external_event(self, event_id: str, processing_sequence: int) -> int:
        if self._external_failure_injector is not None:
            self._external_failure_injector("finalize", event_id)
        return self._transition_external_event(
            event_id,
            from_statuses={ExternalTransactionStatus.PENDING},
            to_status=ExternalTransactionStatus.COMMITTED,
            reason="snapshot_committed",
            processing_sequence=processing_sequence,
        )

    def orphan_external_event(self, event_id: str, reason: str) -> int:
        return self._transition_external_event(
            event_id,
            from_statuses={ExternalTransactionStatus.PENDING},
            to_status=ExternalTransactionStatus.ORPHANED,
            reason=reason,
        )

    def compensate_external_event(self, event_id: str, reason: str) -> int:
        return self._transition_external_event(
            event_id,
            from_statuses={
                ExternalTransactionStatus.PENDING,
                ExternalTransactionStatus.ORPHANED,
            },
            to_status=ExternalTransactionStatus.COMPENSATED,
            reason=reason,
            quarantine=True,
        )

    def _transition_external_event(
        self,
        event_id: str,
        *,
        from_statuses: set[ExternalTransactionStatus],
        to_status: ExternalTransactionStatus,
        reason: str,
        processing_sequence: int | None = None,
        quarantine: bool = False,
    ) -> int:
        result = self.db1.get(
            where={"source_event_id": event_id}, include=["metadatas"]
        )
        changed = 0
        for episode_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            metadata = dict(raw_metadata or {})
            current = ExternalTransactionStatus(
                str(metadata.get("external_transaction_status", "committed"))
            )
            if current == to_status:
                continue
            if current not in from_statuses:
                continue
            if (
                processing_sequence is not None
                and _metadata_int(metadata.get("processing_sequence"), -1)
                != processing_sequence
            ):
                raise ValueError("external transaction processing sequence mismatch")
            revision = (
                _metadata_int(metadata.get("external_transaction_revision"), 1) + 1
            )
            metadata["external_transaction_status"] = to_status.value
            metadata["external_transaction_revision"] = revision
            metadata["external_transaction_updated_at"] = _now_iso()
            audit = _loads_json_dict_list(metadata.get("external_transaction_audit"))
            audit.append(_external_audit_entry(revision, to_status, reason))
            metadata["external_transaction_audit"] = json.dumps(audit)
            if quarantine:
                metadata["lifecycle_status"] = MemoryLifecycleStatus.QUARANTINED.value
                metadata["training_included"] = False
            self.db1.update(ids=[str(episode_id)], metadatas=[metadata])
            changed += 1
        return changed

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
        source_feedback_ids: list[str] | None = None,
        confidence: float = 1.0,
        valid_from: str | None = None,
        valid_until: str | None = None,
        expires_at: str | None = None,
        decay_rate: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not text.strip():
            raise ValueError("Semantic memory text must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Semantic confidence must be between zero and one")
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("Semantic decay rate must be between zero and one")
        _validate_semantic_timestamps(valid_from, valid_until, expires_at)
        content_hash = _semantic_content_hash(text)
        duplicate = self.db2.get(
            where={"content_hash": content_hash}, include=["metadatas"]
        )
        duplicate_metadata = _first_metadata(duplicate)
        duplicate_ids = duplicate.get("ids") or []
        if duplicate_metadata is not None and duplicate_ids:
            duplicate_metadata["source_episode_ids"] = json.dumps(
                _unique_strings(
                    _loads_json_list(duplicate_metadata.get("source_episode_ids"))
                    + (source_episode_ids or [])
                )
            )
            duplicate_metadata["source_feedback_ids"] = json.dumps(
                _unique_strings(
                    _loads_json_list(duplicate_metadata.get("source_feedback_ids"))
                    + (source_feedback_ids or [])
                )
            )
            _append_semantic_audit(
                duplicate_metadata,
                operation="deduplicate",
                detail={"content_hash": content_hash},
            )
            semantic_id = str(duplicate_ids[0])
            self.db2.update(ids=[semantic_id], metadatas=[duplicate_metadata])
            self.reevaluate_semantic_sources(semantic_id)
            return semantic_id
        semantic_id = f"semantic-{uuid4()}"
        now = _now_iso()
        record_metadata: Metadata = {
            "text": text,
            "source_episode_ids": json.dumps(source_episode_ids or []),
            "source_feedback_ids": json.dumps(source_feedback_ids or []),
            "rejected_source_feedback_ids": "[]",
            "record_type": MemoryRecordType.SEMANTIC_MEMORY.value,
            "archived": False,
            "created_at": now,
            "schema_version": 2,
            "version": 1,
            "content_hash": content_hash,
            "confidence": confidence,
            "validity": "valid",
            "valid_from": valid_from or "",
            "valid_until": valid_until or "",
            "expires_at": expires_at or "",
            "decay_rate": decay_rate,
            "last_confirmed_at": now,
            "lifecycle_status": SemanticLifecycleStatus.ACTIVE.value,
            "supersedes_id": "",
            "superseded_by_id": "",
            "corrected_by_id": "",
            "contradiction_ids": "[]",
            "merge_candidate_ids": "[]",
            "audit_log": json.dumps(
                [_semantic_audit_entry("create", {"content_hash": content_hash})]
            ),
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
        self.expire_semantic_records()
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
                and record.external_transaction_status
                == ExternalTransactionStatus.COMMITTED
            ][: self.settings.memory.db1_top_k],
            db2_results=[
                _with_effective_confidence(record)
                for record in semantic
                if _semantic_is_retrievable(record)
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
            and record.external_transaction_status
            == ExternalTransactionStatus.COMMITTED
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
        self.reevaluate_semantics_for_episode(episode_id)
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
        self.reevaluate_semantics_for_episode(episode_id)
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

    def semantic_is_retrievable(self, record: SemanticMemoryRecord) -> bool:
        return (
            _semantic_is_retrievable(record)
            and record.metadata.get("publication_status", "published") == "published"
        )

    def archive_episodic(self, episode_id: str) -> EpisodicMemoryRecord | None:
        result = self.db1.get(ids=[episode_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        metadata["archived"] = True
        self.db1.update(ids=[episode_id], metadatas=[metadata])
        self.reevaluate_semantics_for_episode(episode_id)
        return self.get_episodic(episode_id)

    def archive_semantic(
        self, memory_id: str, *, idempotency_key: str | None = None
    ) -> SemanticMemoryRecord | None:
        result = self.db2.get(ids=[memory_id], include=["metadatas"])
        metadata = _first_metadata(result)
        if metadata is None:
            return None
        if idempotency_key is not None and _semantic_operation_seen(
            metadata, idempotency_key, "archive", {}
        ):
            return self.get_semantic(memory_id)
        if metadata.get("archived") is True:
            if idempotency_key is not None:
                _append_semantic_audit(
                    metadata,
                    operation="archive",
                    detail={},
                    idempotency_key=idempotency_key,
                )
                self.db2.update(ids=[memory_id], metadatas=[metadata])
            return self.get_semantic(memory_id)
        metadata["archived"] = True
        _append_semantic_audit(
            metadata,
            operation="archive",
            detail={},
            idempotency_key=idempotency_key,
        )
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        return self.get_semantic(memory_id)

    def restore_semantic(
        self, memory_id: str, *, idempotency_key: str
    ) -> SemanticMemoryRecord | None:
        metadata = self._semantic_metadata(memory_id)
        if metadata is None:
            return None
        if _semantic_operation_seen(metadata, idempotency_key, "restore", {}):
            return self.get_semantic(memory_id)
        metadata["archived"] = False
        _append_semantic_audit(
            metadata, operation="restore", detail={}, idempotency_key=idempotency_key
        )
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        self.reevaluate_semantic_sources(memory_id)
        return self.get_semantic(memory_id)

    def forget_semantic(
        self, memory_id: str, *, idempotency_key: str
    ) -> SemanticMemoryRecord | None:
        return self._set_semantic_lifecycle(
            memory_id,
            SemanticLifecycleStatus.FORGOTTEN,
            operation="forget",
            idempotency_key=idempotency_key,
        )

    def delete_semantic(self, memory_id: str, *, idempotency_key: str) -> bool:
        metadata = self._semantic_metadata(memory_id)
        if metadata is None:
            return False
        result = self.db2.get(include=["metadatas"])
        for related_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            if related_id == memory_id:
                continue
            related = dict(raw_metadata or {})
            changed = False
            for field in ("contradiction_ids", "merge_candidate_ids"):
                values = _loads_json_list(related.get(field))
                if memory_id in values:
                    related[field] = json.dumps(
                        [value for value in values if value != memory_id]
                    )
                    changed = True
            for field in ("supersedes_id", "superseded_by_id", "corrected_by_id"):
                if related.get(field) == memory_id:
                    related[field] = ""
                    changed = True
            if changed:
                _append_semantic_audit(
                    related,
                    operation="lineage_target_deleted",
                    detail={"target_id": memory_id},
                    idempotency_key=idempotency_key,
                )
                self.db2.update(ids=[str(related_id)], metadatas=[related])
        # Logical forgetting is the reversible path that retains the record's
        # complete content and audit history.
        self.db2.delete(ids=[memory_id])
        return True

    def propose_semantic_relationship(
        self,
        memory_id: str,
        *,
        target_id: str,
        relationship: str,
        idempotency_key: str,
    ) -> SemanticMemoryRecord:
        if relationship not in {"merge", "contradiction", "supersession", "correction"}:
            raise ValueError("Unsupported semantic relationship")
        if memory_id == target_id:
            raise ValueError("Semantic memory cannot relate to itself")
        metadata = self._semantic_metadata(memory_id)
        target = self._semantic_metadata(target_id)
        if metadata is None or target is None:
            raise ValueError("Unknown semantic relationship endpoint")
        detail = {"target_id": target_id, "relationship": relationship}
        if _semantic_operation_seen(metadata, idempotency_key, "relationship", detail):
            record = self.get_semantic(memory_id)
            assert record is not None
            return record
        if relationship == "merge":
            candidates = _loads_json_list(metadata.get("merge_candidate_ids"))
            metadata["merge_candidate_ids"] = json.dumps(
                _unique_strings([*candidates, target_id])
            )
        elif relationship == "contradiction":
            self._link_contradiction(memory_id, metadata, target_id, target)
        else:
            old_id, old_metadata = target_id, target
            new_status = (
                SemanticLifecycleStatus.CORRECTED
                if relationship == "correction"
                else SemanticLifecycleStatus.SUPERSEDED
            )
            old_metadata["lifecycle_status"] = new_status.value
            reverse_field = (
                "corrected_by_id"
                if relationship == "correction"
                else "superseded_by_id"
            )
            old_metadata[reverse_field] = memory_id
            metadata["supersedes_id"] = old_id
            _append_semantic_audit(
                old_metadata,
                operation="relationship_target",
                detail={"source_id": memory_id, "relationship": relationship},
            )
            self.db2.update(ids=[old_id], metadatas=[old_metadata])
        _append_semantic_audit(
            metadata,
            operation="relationship",
            detail=detail,
            idempotency_key=idempotency_key,
        )
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        record = self.get_semantic(memory_id)
        assert record is not None
        return record

    def update_semantic_policy(
        self,
        memory_id: str,
        *,
        idempotency_key: str,
        confidence: float,
        validity: str,
        valid_from: str | None,
        valid_until: str | None,
        expires_at: str | None,
        decay_rate: float,
    ) -> SemanticMemoryRecord | None:
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= decay_rate <= 1.0:
            raise ValueError(
                "Semantic confidence and decay rate must be between zero and one"
            )
        if validity not in {"valid", "disputed", "invalid"}:
            raise ValueError("Unsupported semantic validity")
        _validate_semantic_timestamps(valid_from, valid_until, expires_at)
        metadata = self._semantic_metadata(memory_id)
        if metadata is None:
            return None
        detail = {
            "confidence": confidence,
            "validity": validity,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "expires_at": expires_at,
            "decay_rate": decay_rate,
        }
        if _semantic_operation_seen(metadata, idempotency_key, "policy", detail):
            return self.get_semantic(memory_id)
        metadata.update(
            {
                "confidence": confidence,
                "validity": validity,
                "valid_from": valid_from or "",
                "valid_until": valid_until or "",
                "expires_at": expires_at or "",
                "decay_rate": decay_rate,
                "last_confirmed_at": _now_iso(),
            }
        )
        if metadata.get("lifecycle_status") == SemanticLifecycleStatus.EXPIRED.value:
            metadata["lifecycle_status"] = SemanticLifecycleStatus.ACTIVE.value
        _append_semantic_audit(
            metadata,
            operation="policy",
            detail=detail,
            idempotency_key=idempotency_key,
        )
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        return self.reevaluate_semantic_sources(memory_id)

    def semantic_graph(self, memory_id: str) -> list[SemanticMemoryRecord]:
        root = self.get_semantic(memory_id)
        if root is None:
            return []
        records: list[SemanticMemoryRecord] = []
        pending = [root]
        seen: set[str] = set()
        while pending:
            record = pending.pop(0)
            if record.id in seen:
                continue
            seen.add(record.id)
            records.append(record)
            related = {
                *record.contradiction_ids,
                *record.merge_candidate_ids,
                *([record.supersedes_id] if record.supersedes_id else []),
                *([record.superseded_by_id] if record.superseded_by_id else []),
                *([record.corrected_by_id] if record.corrected_by_id else []),
            }
            for related_id in sorted(related - seen):
                related_record = self.get_semantic(related_id)
                if related_record is not None:
                    pending.append(related_record)
        return records

    def reevaluate_semantics_for_episode(self, episode_id: str) -> None:
        result = self.db2.get(include=["metadatas"])
        for semantic_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            metadata = dict(raw_metadata or {})
            if episode_id in _loads_json_list(metadata.get("source_episode_ids")):
                self.reevaluate_semantic_sources(str(semantic_id))

    def reevaluate_semantic_sources(
        self, memory_id: str
    ) -> SemanticMemoryRecord | None:
        metadata = self._semantic_metadata(memory_id)
        if metadata is None:
            return None
        source_ids = _loads_json_list(metadata.get("source_episode_ids"))
        feedback_ids = _loads_json_list(metadata.get("source_feedback_ids"))
        rejected_feedback_ids = set(
            _loads_json_list(metadata.get("rejected_source_feedback_ids"))
        )
        if not source_ids and not feedback_ids:
            return self.get_semantic(memory_id)
        viable = any(item not in rejected_feedback_ids for item in feedback_ids)
        for source_id in source_ids:
            source = self.get_episodic(source_id)
            if (
                source is not None
                and source.lifecycle_status == MemoryLifecycleStatus.ACTIVE
                and source.validation_status != ValidationStatus.REJECTED
            ):
                viable = True
                break
        current = str(metadata.get("lifecycle_status", "active"))
        if not viable and current == SemanticLifecycleStatus.ACTIVE.value:
            metadata["lifecycle_status"] = SemanticLifecycleStatus.SOURCE_REJECTED.value
            _append_semantic_audit(
                metadata, operation="source_reevaluation", detail={"viable": False}
            )
        elif viable and current == SemanticLifecycleStatus.SOURCE_REJECTED.value:
            metadata["lifecycle_status"] = SemanticLifecycleStatus.ACTIVE.value
            _append_semantic_audit(
                metadata, operation="source_reevaluation", detail={"viable": True}
            )
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        return self.get_semantic(memory_id)

    def reevaluate_semantics_for_feedback(
        self, feedback_id: str, *, rejected: bool
    ) -> None:
        result = self.db2.get(include=["metadatas"])
        for semantic_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            metadata = dict(raw_metadata or {})
            if feedback_id not in _loads_json_list(metadata.get("source_feedback_ids")):
                continue
            rejected_ids = _loads_json_list(
                metadata.get("rejected_source_feedback_ids")
            )
            if rejected:
                rejected_ids = _unique_strings([*rejected_ids, feedback_id])
            else:
                rejected_ids = [item for item in rejected_ids if item != feedback_id]
            if rejected_ids == _loads_json_list(
                metadata.get("rejected_source_feedback_ids")
            ):
                continue
            metadata["rejected_source_feedback_ids"] = json.dumps(rejected_ids)
            _append_semantic_audit(
                metadata,
                operation="feedback_source_status",
                detail={"feedback_id": feedback_id, "rejected": rejected},
            )
            self.db2.update(ids=[str(semantic_id)], metadatas=[metadata])
            self.reevaluate_semantic_sources(str(semantic_id))

    def expire_semantic_records(self) -> None:
        result = self.db2.get(include=["metadatas"])
        now = datetime.now(UTC)
        for semantic_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            metadata = dict(raw_metadata or {})
            if metadata.get("lifecycle_status", "active") != "active":
                continue
            expires_at = _parse_timestamp(_optional_str(metadata.get("expires_at")))
            valid_until = _parse_timestamp(_optional_str(metadata.get("valid_until")))
            confidence = float(str(metadata.get("confidence", 1.0)))
            decay_rate = float(str(metadata.get("decay_rate", 0.0)))
            confirmed = _parse_timestamp(
                str(metadata.get("last_confirmed_at", metadata.get("created_at", "")))
            )
            decayed = confidence
            if confirmed is not None and decay_rate > 0.0:
                elapsed_days = max(0.0, (now - confirmed).total_seconds() / 86400)
                decayed *= (1.0 - decay_rate) ** elapsed_days
            reason = None
            if expires_at is not None and expires_at <= now:
                reason = "expiry"
            elif valid_until is not None and valid_until <= now:
                reason = "validity_end"
            elif decayed <= 0.01:
                reason = "confidence_decay"
            if reason is None:
                continue
            metadata["lifecycle_status"] = SemanticLifecycleStatus.EXPIRED.value
            _append_semantic_audit(
                metadata,
                operation="expire",
                detail={"reason": reason},
            )
            self.db2.update(ids=[str(semantic_id)], metadatas=[metadata])

    def _semantic_metadata(self, memory_id: str) -> dict[str, Any] | None:
        return _first_metadata(self.db2.get(ids=[memory_id], include=["metadatas"]))

    def _set_semantic_lifecycle(
        self,
        memory_id: str,
        status: SemanticLifecycleStatus,
        *,
        operation: str,
        idempotency_key: str,
    ) -> SemanticMemoryRecord | None:
        metadata = self._semantic_metadata(memory_id)
        if metadata is None:
            return None
        detail = {"status": status.value}
        if _semantic_operation_seen(metadata, idempotency_key, operation, detail):
            return self.get_semantic(memory_id)
        metadata["lifecycle_status"] = status.value
        _append_semantic_audit(
            metadata,
            operation=operation,
            detail=detail,
            idempotency_key=idempotency_key,
        )
        self.db2.update(ids=[memory_id], metadatas=[metadata])
        return self.get_semantic(memory_id)

    def _link_contradiction(
        self,
        memory_id: str,
        metadata: dict[str, Any],
        target_id: str,
        target: dict[str, Any],
    ) -> None:
        metadata["contradiction_ids"] = json.dumps(
            _unique_strings(
                [*_loads_json_list(metadata.get("contradiction_ids")), target_id]
            )
        )
        target["contradiction_ids"] = json.dumps(
            _unique_strings(
                [*_loads_json_list(target.get("contradiction_ids")), memory_id]
            )
        )
        _append_semantic_audit(
            target,
            operation="relationship_target",
            detail={"source_id": memory_id, "relationship": "contradiction"},
        )
        self.db2.update(ids=[target_id], metadatas=[target])

    def _backfill_semantic_records(self) -> None:
        result = self.db2.get(include=["documents", "metadatas"])
        for semantic_id, document, raw_metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
            strict=False,
        ):
            metadata = dict(raw_metadata or {})
            if int(str(metadata.get("schema_version", 1))) >= 2:
                continue
            text = str(metadata.get("text", document or ""))
            now = str(metadata.get("created_at", "")) or _now_iso()
            metadata.update(
                {
                    "schema_version": 2,
                    "version": 1,
                    "content_hash": _semantic_content_hash(text),
                    "confidence": 1.0,
                    "validity": "valid",
                    "valid_from": "",
                    "valid_until": "",
                    "expires_at": "",
                    "decay_rate": 0.0,
                    "last_confirmed_at": now,
                    "lifecycle_status": "active",
                    "supersedes_id": "",
                    "superseded_by_id": "",
                    "corrected_by_id": "",
                    "contradiction_ids": "[]",
                    "source_feedback_ids": "[]",
                    "rejected_source_feedback_ids": "[]",
                    "merge_candidate_ids": "[]",
                    "audit_log": json.dumps([_semantic_audit_entry("backfill", {})]),
                }
            )
            self.db2.update(ids=[str(semantic_id)], metadatas=[metadata])

    def _backfill_episodic_transactions(self) -> None:
        result = self.db1.get(include=["metadatas"])
        for episode_id, raw_metadata in zip(
            result.get("ids") or [], result.get("metadatas") or [], strict=False
        ):
            metadata = dict(raw_metadata or {})
            if "external_transaction_status" in metadata:
                continue
            created_at = str(metadata.get("created_at", "")) or _now_iso()
            metadata.update(
                {
                    "external_transaction_id": f"external-{episode_id}",
                    "external_transaction_schema_version": 1,
                    "external_transaction_revision": 1,
                    "external_transaction_status": ExternalTransactionStatus.COMMITTED.value,
                    "external_transaction_updated_at": created_at,
                    "external_transaction_audit": json.dumps(
                        [
                            _external_audit_entry(
                                1,
                                ExternalTransactionStatus.COMMITTED,
                                "legacy_backfill",
                            )
                        ]
                    ),
                }
            )
            self.db1.update(ids=[str(episode_id)], metadatas=[metadata])

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


def _episodic_document(user_input: str, response: str) -> str:
    return f"User: {user_input}\nAssistant: {response}".strip()


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
        external_transaction_id=str(
            metadata.get("external_transaction_id", f"external-{record_id}")
        ),
        external_transaction_status=ExternalTransactionStatus(
            str(
                metadata.get(
                    "external_transaction_status",
                    ExternalTransactionStatus.COMMITTED.value,
                )
            )
        ),
        external_transaction_revision=int(
            metadata.get("external_transaction_revision", 1)
        ),
    )


def _semantic_record_from_metadata(
    record_id: str, document: str, metadata: dict[str, Any]
) -> SemanticMemoryRecord:
    extra = _loads_json_dict(metadata.get("extra"))
    lifecycle = SemanticLifecycleStatus(
        str(metadata.get("lifecycle_status", SemanticLifecycleStatus.ACTIVE.value))
    )
    expires_at = _optional_str(metadata.get("expires_at"))
    if lifecycle == SemanticLifecycleStatus.ACTIVE and _timestamp_has_passed(
        expires_at
    ):
        lifecycle = SemanticLifecycleStatus.EXPIRED
    record = SemanticMemoryRecord(
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
        schema_version=int(metadata.get("schema_version", 1)),
        version=int(metadata.get("version", 1)),
        content_hash=str(
            metadata.get("content_hash", _semantic_content_hash(document))
        ),
        confidence=float(metadata.get("confidence", 1.0)),
        validity=str(metadata.get("validity", "valid")),
        valid_from=_optional_str(metadata.get("valid_from")),
        valid_until=_optional_str(metadata.get("valid_until")),
        expires_at=expires_at,
        decay_rate=float(metadata.get("decay_rate", 0.0)),
        last_confirmed_at=str(
            metadata.get("last_confirmed_at", metadata.get("created_at", ""))
        ),
        lifecycle_status=lifecycle,
        supersedes_id=_optional_str(metadata.get("supersedes_id")),
        superseded_by_id=_optional_str(metadata.get("superseded_by_id")),
        corrected_by_id=_optional_str(metadata.get("corrected_by_id")),
        contradiction_ids=_loads_json_list(metadata.get("contradiction_ids")),
        source_feedback_ids=_loads_json_list(metadata.get("source_feedback_ids")),
        merge_candidate_ids=_loads_json_list(metadata.get("merge_candidate_ids")),
        audit_log=_loads_json_dict_list(metadata.get("audit_log")),
    )
    return _with_effective_confidence(record)


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


def _loads_json_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str):
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [dict(item) for item in loaded if isinstance(item, dict)]


def _content_hash(user_input: str, response: str) -> str:
    normalized = json.dumps([user_input.strip(), response.strip()], ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _dedup_key(source_event_id: str | None, content_hash: str) -> str:
    return f"{source_event_id or 'content'}:{content_hash}"


def _semantic_content_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _semantic_audit_entry(
    operation: str,
    detail: dict[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    return {
        "event_id": f"memory-audit-{uuid4()}",
        "operation": operation,
        "detail": detail,
        "idempotency_key": idempotency_key,
        "created_at": _now_iso(),
    }


def _append_semantic_audit(
    metadata: dict[str, Any],
    *,
    operation: str,
    detail: dict[str, Any],
    idempotency_key: str | None = None,
) -> None:
    audit = _loads_json_dict_list(metadata.get("audit_log"))
    audit.append(_semantic_audit_entry(operation, detail, idempotency_key))
    metadata["audit_log"] = json.dumps(audit)
    metadata["version"] = int(metadata.get("version", 1)) + 1


def _semantic_operation_seen(
    metadata: dict[str, Any],
    idempotency_key: str,
    operation: str,
    detail: dict[str, Any],
) -> bool:
    for event in _loads_json_dict_list(metadata.get("audit_log")):
        if event.get("idempotency_key") != idempotency_key:
            continue
        if event.get("operation") != operation or event.get("detail") != detail:
            raise ValueError("Idempotency key was already used for another operation")
        return True
    return False


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _timestamp_has_passed(value: str | None) -> bool:
    parsed = _parse_timestamp(value)
    return parsed is not None and parsed <= datetime.now(UTC)


def _validate_semantic_timestamps(
    valid_from: str | None, valid_until: str | None, expires_at: str | None
) -> None:
    parsed = [
        _parse_timestamp(value) for value in (valid_from, valid_until, expires_at)
    ]
    for value, timestamp in zip(
        (valid_from, valid_until, expires_at), parsed, strict=True
    ):
        if value and timestamp is None:
            raise ValueError("Semantic validity timestamps must be ISO-8601 values")
    if parsed[0] is not None and parsed[1] is not None and parsed[0] >= parsed[1]:
        raise ValueError("Semantic valid_from must precede valid_until")


def _with_effective_confidence(record: SemanticMemoryRecord) -> SemanticMemoryRecord:
    confirmed = _parse_timestamp(record.last_confirmed_at)
    if confirmed is None or record.decay_rate == 0.0:
        effective = record.confidence
    else:
        elapsed_days = max(0.0, (datetime.now(UTC) - confirmed).total_seconds() / 86400)
        effective = record.confidence * ((1.0 - record.decay_rate) ** elapsed_days)
    return replace(record, effective_confidence=max(0.0, min(1.0, effective)))


def _semantic_is_retrievable(record: SemanticMemoryRecord) -> bool:
    now = datetime.now(UTC)
    valid_from = _parse_timestamp(record.valid_from)
    valid_until = _parse_timestamp(record.valid_until)
    expires_at = _parse_timestamp(record.expires_at)
    return (
        not record.archived
        and record.lifecycle_status == SemanticLifecycleStatus.ACTIVE
        and record.validity == "valid"
        and (valid_from is None or valid_from <= now)
        and (valid_until is None or valid_until > now)
        and (expires_at is None or expires_at > now)
        and record.effective_confidence > 0.01
    )


def _optional_str(value: Any) -> str | None:
    return None if value in {None, ""} else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _external_sequence(primary: Any, fallback: Any) -> int | None:
    value = primary if primary not in {None, -1, "-1"} else fallback
    return None if value is None else int(value)


def _metadata_int(value: Any, default: int) -> int:
    return default if value is None else int(str(value))


def _external_audit_entry(
    revision: int, status: ExternalTransactionStatus, reason: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "revision": revision,
        "status": status.value,
        "timestamp": _now_iso(),
        "reason": reason,
    }


def _reject_private_transaction_metadata(value: Any) -> None:
    forbidden = {
        "attachmentbody",
        "credential",
        "credentials",
        "hiddenthought",
        "password",
        "prompt",
        "rawprompt",
        "secret",
        "token",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(
                character for character in str(key).lower() if character.isalnum()
            )
            if any(marker in normalized for marker in forbidden):
                raise ValueError(
                    "private field is forbidden in external transaction metadata"
                )
            _reject_private_transaction_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private_transaction_metadata(item)


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

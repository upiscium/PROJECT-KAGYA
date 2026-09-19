"""ChromaDB-backed dual memory implementation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.types import Metadata

from kagya.config import Settings
from kagya.identifiers import validate_identifier
from kagya.memory.consolidation import build_consolidation_prompt
from kagya.memory.memory_evaluator import MemoryEvaluator
from kagya.memory.memory_schema import (
    EpisodicMemoryRecord,
    MemoryContext,
    MemoryRecordType,
    SemanticMemoryRecord,
)
from kagya.models import ModelProvider
from kagya.privacy import PRIVATE_FIELD_KEYS, normalize_private_key, reject_private_fields, scrub_private_fields


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


def _resolve_embedding_function(embedding_function: Any | None) -> Any:
    """Return the injected embedding function or the deterministic baseline adapter."""

    if embedding_function is None:
        return DeterministicEmbeddingFunction()
    return embedding_function


class EpisodicMemoryReadError(Exception):
    """A bounded failure to read committed episodic Memory."""


class EpisodicMemoryFormatError(Exception):
    """Committed episodic Memory has malformed domain content."""


class SemanticMemoryReadError(Exception):
    """A bounded failure to read committed semantic Memory."""


class SemanticMemoryFormatError(Exception):
    """Committed semantic Memory has malformed domain content."""


@dataclass(frozen=True, slots=True)
class CommittedEpisodicMemory:
    """One committed DB1 document and its parsed metadata projection."""

    document: str
    metadata: dict[str, Any]
    record: EpisodicMemoryRecord


@dataclass(frozen=True, slots=True)
class CommittedSemanticMemory:
    """One committed DB2 document and its parsed metadata projection."""

    document: str
    metadata: dict[str, Any]
    record: SemanticMemoryRecord


class DualMemorySystem:
    """Dual memory backed by DB1 hippocampus and DB2 cortex Chroma collections."""

    def __init__(
        self,
        settings: Settings,
        embedding_function: Any | None = None,
        evaluator: MemoryEvaluator | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_function = _resolve_embedding_function(embedding_function)
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
        self._scrub_legacy_private_records()

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
    ) -> str:
        extra_metadata = metadata or {}
        reject_private_fields(extra_metadata, context="Episodic memory metadata")
        episode_id = f"episode-{uuid4()}"
        self._add_episodic(
            episode_id,
            user_input,
            response,
            loss=loss,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            record_type=record_type,
            created_at=_now_iso(),
            metadata=extra_metadata,
        )
        return episode_id

    def get_episodic_record(self, episode_id: str) -> EpisodicMemoryRecord | None:
        """Return one committed DB1 record without consulting pending staging."""

        committed = self.get_committed_episodic(episode_id)
        return None if committed is None else committed.record

    def get_committed_episodic(
        self, episode_id: str
    ) -> CommittedEpisodicMemory | None:
        """Read one DB1 document and metadata together without repairing either."""

        try:
            result = self.db1.get(
                ids=[episode_id], include=["documents", "metadatas"]
            )
        except Exception:
            raise EpisodicMemoryReadError(
                "Committed episodic Memory is unavailable"
            ) from None
        try:
            ids, documents, metadatas = _strict_get_parts(result)
            if not ids:
                return None
            if not isinstance(ids[0], str) or ids[0] != episode_id:
                raise ValueError
            metadata = _strict_metadata(metadatas[0])
            record = _committed_episodic_record(episode_id, documents[0], metadata)
        except Exception:
            raise EpisodicMemoryFormatError(
                "Committed episodic Memory is invalid"
            ) from None
        return CommittedEpisodicMemory(
            document=documents[0], metadata=metadata, record=record
        )

    def get_committed_semantic(
        self, semantic_id: str
    ) -> CommittedSemanticMemory | None:
        """Read exactly one DB2 document and metadata without repairing it."""

        try:
            result = self.db2.get(ids=[semantic_id], include=["documents", "metadatas"])
        except Exception:
            raise SemanticMemoryReadError(
                "Committed semantic Memory is unavailable"
            ) from None
        try:
            ids, documents, metadatas = _strict_get_parts(result)
            if not ids:
                return None
            if not isinstance(ids[0], str) or ids[0] != semantic_id:
                raise ValueError
            metadata = _strict_metadata(metadatas[0])
            record = _committed_semantic_record(semantic_id, documents[0], metadata)
        except Exception:
            raise SemanticMemoryFormatError(
                "Committed semantic Memory is invalid"
            ) from None
        return CommittedSemanticMemory(
            document=documents[0], metadata=metadata, record=record
        )

    def publish_coordinated_episodic(
        self,
        episode_id: str,
        user_input: str,
        response: str,
        *,
        loss: float,
        emotion_valence: float,
        emotion_arousal: float,
        record_type: MemoryRecordType,
        created_at: str,
        coordination_schema: int | None = None,
        context_id: str | None = None,
        source_channel: str | None = None,
        source_session_id: str | None = None,
    ) -> None:
        """Publish one already-validated deterministic coordinated record to DB1."""

        self._add_episodic(
            episode_id,
            user_input,
            response,
            loss=loss,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            record_type=record_type,
            created_at=created_at,
            metadata={},
            coordinated=True,
            coordination_schema=coordination_schema,
            context_id=context_id,
            source_channel=source_channel,
            source_session_id=source_session_id,
        )

    def _add_episodic(
        self,
        episode_id: str,
        user_input: str,
        response: str,
        *,
        loss: float,
        emotion_valence: float,
        emotion_arousal: float,
        record_type: MemoryRecordType,
        created_at: str,
        metadata: Mapping[str, Any],
        coordinated: bool = False,
        coordination_schema: int | None = None,
        context_id: str | None = None,
        source_channel: str | None = None,
        source_session_id: str | None = None,
    ) -> None:
        record_metadata = canonical_episodic_metadata(
            user_input,
            response,
            loss=loss,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            record_type=record_type,
            created_at=created_at,
            metadata=metadata,
            coordinated=coordinated,
            coordination_schema=coordination_schema,
            context_id=context_id,
            source_channel=source_channel,
            source_session_id=source_session_id,
        )
        self.db1.add(
            ids=[episode_id],
            documents=[canonical_episodic_document(user_input, response)],
            metadatas=[record_metadata],
        )

    def save_semantic(
        self,
        text: str,
        *,
        source_episode_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        extra_metadata = metadata or {}
        reject_private_fields(extra_metadata, context="Semantic memory metadata")
        semantic_id = f"semantic-{uuid4()}"
        record_metadata: Metadata = {
            "text": text,
            "source_episode_ids": json.dumps(source_episode_ids or []),
            "record_type": MemoryRecordType.SEMANTIC_MEMORY.value,
            "created_at": _now_iso(),
            "extra": json.dumps(extra_metadata),
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

    def _scrub_legacy_private_records(self) -> None:
        """One-way sanitize pre-R02 records before they can be retrieved again."""

        episodic = self.db1.get(include=["documents", "metadatas"])
        episodic_ids = episodic.get("ids") or []
        episodic_documents = episodic.get("documents") or []
        episodic_metadatas = episodic.get("metadatas") or []
        for record_id, document, raw_metadata in zip(
            episodic_ids,
            episodic_documents,
            episodic_metadatas,
            strict=False,
        ):
            metadata = dict(raw_metadata or {})
            sanitized = _sanitize_persisted_metadata(metadata)
            visible_document = canonical_episodic_document(
                str(sanitized.get("user_input", "")),
                str(sanitized.get("response", "")),
            )
            if _is_coordinated_episodic_metadata(metadata):
                try:
                    _committed_episodic_record(str(record_id), document, metadata)
                except ValueError:
                    raise EpisodicMemoryFormatError(
                        "Committed episodic Memory is invalid"
                    ) from None
                if sanitized != metadata or document != visible_document:
                    raise EpisodicMemoryFormatError(
                        "Committed episodic Memory is invalid"
                    )
                continue
            if sanitized == metadata and document == visible_document:
                continue
            self.db1.delete(ids=[str(record_id)])
            self.db1.add(
                ids=[str(record_id)],
                documents=[visible_document],
                metadatas=[sanitized],
            )

        semantic = self.db2.get(include=["metadatas"])
        semantic_ids = semantic.get("ids") or []
        semantic_metadatas = semantic.get("metadatas") or []
        for record_id, raw_metadata in zip(
            semantic_ids, semantic_metadatas, strict=False
        ):
            metadata = dict(raw_metadata or {})
            sanitized = _sanitize_persisted_metadata(metadata)
            if sanitized != metadata:
                self.db2.update(ids=[str(record_id)], metadatas=[sanitized])


def _embed_text(text: str) -> list[float]:
    buckets = [0.0] * 16
    for index, char in enumerate(text):
        buckets[index % len(buckets)] += float(ord(char) % 31) / 31.0
    magnitude = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [value / magnitude for value in buckets]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def canonical_episodic_document(user_input: str, response: str) -> str:
    """Build the authoritative visible DB1 episodic document."""

    return f"User: {user_input}\nAssistant: {response}".strip()


def canonical_episodic_metadata(
    user_input: str,
    response: str,
    *,
    loss: float,
    emotion_valence: float,
    emotion_arousal: float,
    record_type: MemoryRecordType,
    created_at: str,
    metadata: Mapping[str, Any],
    coordinated: bool = False,
    coordination_schema: int | None = None,
    context_id: str | None = None,
    source_channel: str | None = None,
    source_session_id: str | None = None,
) -> Metadata:
    """Build the authoritative DB1 metadata map for one episodic record."""

    result: dict[str, str | int | float | bool] = {
        "user_input": user_input,
        "response": response,
        "loss": float(loss),
        "emotion_valence": float(emotion_valence),
        "emotion_arousal": float(emotion_arousal),
        "record_type": record_type.value,
        "archived": False,
        "created_at": created_at,
        "extra": json.dumps(dict(metadata)),
    }
    if not coordinated:
        if any(
            value is not None
            for value in (context_id, source_channel, source_session_id)
        ):
            raise ValueError("Uncoordinated episodic records do not support provenance")
        return result
    if coordination_schema is None:
        coordination_schema = (
            2
            if any(
                value is not None
                for value in (context_id, source_channel, source_session_id)
            )
            else 1
        )
    if coordinated:
        if type(coordination_schema) is not int or coordination_schema not in (1, 2):
            raise ValueError("unsupported coordination schema")
        if coordination_schema == 2:
            _validate_provenance(context_id, source_channel, source_session_id)
        elif any(value is not None for value in (context_id, source_channel, source_session_id)):
            raise ValueError("schema 1 does not support provenance")
        result["coordination_schema"] = coordination_schema
        if coordination_schema == 2:
            for key, value in (
                ("context_id", context_id),
                ("source_channel", source_channel),
                ("source_session_id", source_session_id),
            ):
                if value is not None:
                    result[key] = value
    return result


def _is_coordinated_episodic_metadata(metadata: Mapping[str, Any]) -> bool:
    return "coordination_schema" in metadata or any(
        key in metadata for key in ("context_id", "source_channel", "source_session_id")
    )


def _sanitize_persisted_metadata(metadata: Mapping[str, Any]) -> Metadata:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if normalize_private_key(key) in PRIVATE_FIELD_KEYS:
            continue
        if key == "extra":
            sanitized[key] = _sanitize_extra_metadata(value)
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
    return sanitized


def _sanitize_extra_metadata(value: Any) -> str:
    if not isinstance(value, str):
        return "{}"
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return "{}"
    if not isinstance(loaded, dict):
        return "{}"
    sanitized = scrub_private_fields(loaded)
    return json.dumps(sanitized)


def _validate_opaque_id(value: Any) -> str:
    """Validate one shared opaque identifier."""

    return validate_identifier(value)


def _validate_provenance(
    context_id: Any, source_channel: Any, source_session_id: Any
) -> None:
    values = (context_id, source_channel, source_session_id)
    if all(value is None for value in values):
        return
    if context_id is None or source_channel is None:
        raise ValueError
    for value in values:
        if value is not None:
            _validate_opaque_id(value)


def _provenance_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[int | None, str | None, str | None, str | None]:
    provenance_keys = ("context_id", "source_channel", "source_session_id")
    if "coordination_schema" not in metadata:
        if any(key in metadata for key in provenance_keys):
            raise ValueError
        return None, None, None, None
    coordination_schema = metadata["coordination_schema"]
    if type(coordination_schema) is not int or coordination_schema not in (1, 2):
        raise ValueError
    if coordination_schema == 1:
        if any(key in metadata for key in provenance_keys):
            raise ValueError
        return 1, None, None, None
    values = tuple(metadata.get(key) for key in provenance_keys)
    if any(key in metadata and type(metadata[key]) is not str for key in provenance_keys):
        raise ValueError
    _validate_provenance(*values)
    return (2, values[0], values[1], values[2])


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


def _strict_get_parts(
    result: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    ids = result.get("ids")
    documents = result.get("documents")
    metadatas = result.get("metadatas")
    if (
        not isinstance(ids, list)
        or not isinstance(documents, list)
        or not isinstance(metadatas, list)
    ):
        raise ValueError
    if not ids and not documents and not metadatas:
        return [], [], []
    if not (len(ids) == len(documents) == len(metadatas) == 1):
        raise ValueError
    return ids, documents, metadatas


def _strict_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError
    return dict(value)


def _committed_episodic_record(
    record_id: str, document: Any, metadata: dict[str, Any]
) -> EpisodicMemoryRecord:
    if not isinstance(document, str):
        raise ValueError
    required = (
        "user_input",
        "response",
        "loss",
        "emotion_valence",
        "emotion_arousal",
        "record_type",
        "archived",
        "created_at",
        "extra",
    )
    if any(key not in metadata for key in required):
        raise ValueError
    (
        coordination_schema,
        context_id,
        source_channel,
        source_session_id,
    ) = _provenance_from_metadata(metadata)
    if not all(
        isinstance(metadata[key], str)
        for key in ("user_input", "response", "record_type", "created_at", "extra")
    ):
        raise ValueError
    if not all(
        type(metadata[key]) in (int, float)
        for key in ("loss", "emotion_valence", "emotion_arousal")
    ):
        raise ValueError
    if not all(
        math.isfinite(float(metadata[key]))
        for key in ("loss", "emotion_valence", "emotion_arousal")
    ):
        raise ValueError
    if type(metadata["archived"]) is not bool:
        raise ValueError
    if metadata["record_type"] not in (
        MemoryRecordType.EPISODIC_LOG.value,
        MemoryRecordType.THOUGHT_LOG.value,
        MemoryRecordType.EXTRACTED_FACT.value,
        MemoryRecordType.EVALUATION_LOG.value,
    ):
        raise ValueError
    if (
        canonical_episodic_document(metadata["user_input"], metadata["response"])
        != document
    ):
        raise ValueError
    extra = json.loads(metadata["extra"])
    if not isinstance(extra, dict):
        raise ValueError
    reject_private_fields(extra, context="Committed episodic Memory metadata")
    return EpisodicMemoryRecord(
        id=record_id,
        user_input=metadata["user_input"],
        response=metadata["response"],
        loss=float(metadata["loss"]),
        emotion_valence=float(metadata["emotion_valence"]),
        emotion_arousal=float(metadata["emotion_arousal"]),
        record_type=MemoryRecordType(metadata["record_type"]),
        archived=metadata["archived"],
        created_at=metadata["created_at"],
        metadata=extra,
        context_id=context_id,
        source_channel=source_channel,
        source_session_id=source_session_id,
        coordination_schema=coordination_schema,
    )


def _committed_semantic_record(
    record_id: str, document: Any, metadata: dict[str, Any]
) -> SemanticMemoryRecord:
    required = ("text", "source_episode_ids", "record_type", "created_at", "extra")
    if any(key not in metadata for key in required) or not isinstance(document, str):
        raise ValueError
    if not all(isinstance(metadata[key], str) for key in required):
        raise ValueError
    if metadata["record_type"] != MemoryRecordType.SEMANTIC_MEMORY.value:
        raise ValueError
    if metadata["text"] != document:
        raise ValueError
    source_episode_ids = json.loads(metadata["source_episode_ids"])
    extra = json.loads(metadata["extra"])
    if (
        not isinstance(source_episode_ids, list)
        or not all(isinstance(item, str) for item in source_episode_ids)
        or not isinstance(extra, dict)
    ):
        raise ValueError
    return SemanticMemoryRecord(
        id=record_id,
        text=metadata["text"],
        source_episode_ids=source_episode_ids,
        record_type=MemoryRecordType.SEMANTIC_MEMORY,
        created_at=metadata["created_at"],
        metadata=extra,
    )


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
    try:
        (
            coordination_schema,
            context_id,
            source_channel,
            source_session_id,
        ) = _provenance_from_metadata(metadata)
    except ValueError:
        raise EpisodicMemoryFormatError(
            "Committed episodic Memory is invalid"
        ) from None
    return EpisodicMemoryRecord(
        id=record_id,
        user_input=str(metadata.get("user_input", "")),
        response=str(metadata.get("response", "")),
        loss=float(metadata.get("loss", 0.0)),
        emotion_valence=float(metadata.get("emotion_valence", 0.0)),
        emotion_arousal=float(metadata.get("emotion_arousal", 0.0)),
        record_type=MemoryRecordType(
            str(metadata.get("record_type", MemoryRecordType.EPISODIC_LOG.value))
        ),
        archived=bool(metadata.get("archived", False)),
        created_at=str(metadata.get("created_at", "")),
        metadata=_loads_json_dict(metadata.get("extra")),
        context_id=context_id,
        source_channel=source_channel,
        source_session_id=source_session_id,
        coordination_schema=coordination_schema,
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

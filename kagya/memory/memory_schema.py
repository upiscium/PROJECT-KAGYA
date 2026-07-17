"""Data model for logical memory records."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryRecordType(StrEnum):
    EPISODIC_LOG = "episodic_log"
    THOUGHT_LOG = "thought_log"
    EXTRACTED_FACT = "extracted_fact"
    SEMANTIC_MEMORY = "semantic_memory"
    EVALUATION_LOG = "evaluation_log"


class MemoryRecordKind(StrEnum):
    OBSERVATION = "observation"
    EXTERNAL_CLAIM = "external_claim"
    AGENT_INFERENCE = "agent_inference"
    AGENT_ACTION = "agent_action"
    GENERATED_RESPONSE = "generated_response"


class ValidationStatus(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REJECTED = "rejected"


class MemoryLifecycleStatus(StrEnum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    CORRECTED = "corrected"


class ConsolidationStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class GenerationHealth:
    healthy: bool = True
    reasons: list[str] = field(default_factory=list)
    truncated: bool = False
    repetitive: bool = False
    parser_failure: bool = False
    prompt_leakage: bool = False
    non_finite_score: bool = False
    fallback_used: bool = False


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    id: str
    user_input: str
    response: str
    hidden_thought: str = ""
    loss: float = 0.0
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    record_type: MemoryRecordType = MemoryRecordType.EPISODIC_LOG
    archived: bool = False
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    operator_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 3
    input_kind: MemoryRecordKind = MemoryRecordKind.EXTERNAL_CLAIM
    response_kind: MemoryRecordKind = MemoryRecordKind.GENERATED_RESPONSE
    input_confidence: float = 0.7
    response_confidence: float = 0.5
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    lifecycle_status: MemoryLifecycleStatus = MemoryLifecycleStatus.ACTIVE
    generation_health: GenerationHealth = field(default_factory=GenerationHealth)
    content_hash: str = ""
    dedup_key: str = ""
    source_event_id: str | None = None
    source: str = "unknown"
    source_channel: str = "unknown"
    source_session_id: str | None = None
    processing_sequence: int | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    context_id: str | None = None
    provider: str = "unknown"
    model_id: str = "unknown"
    model_revision: str = "unknown"
    adapter_id: str | None = None
    contradiction_ids: list[str] = field(default_factory=list)
    supersedes_id: str | None = None
    corrected_by_id: str | None = None
    consolidation_status: ConsolidationStatus = ConsolidationStatus.PENDING
    consolidation_version: str = ""
    consolidation_attempt_id: str | None = None
    semantic_relevance: float = 0.0
    context_compatibility: float = 0.0
    context_relation: str = "legacy_unknown"
    cross_context: bool = False
    experience_id: str | None = None
    subjective_salience: float = 0.0
    autobiographical_importance: float = 0.0


@dataclass(frozen=True)
class SemanticMemoryRecord:
    id: str
    text: str
    source_episode_ids: list[str] = field(default_factory=list)
    record_type: MemoryRecordType = MemoryRecordType.SEMANTIC_MEMORY
    archived: bool = False
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    operator_metadata: dict[str, Any] = field(default_factory=dict)
    context_id: str | None = None
    source: str = "unknown"
    source_channel: str = "unknown"
    source_session_id: str | None = None
    semantic_relevance: float = 0.0
    context_compatibility: float = 0.0
    context_relation: str = "legacy_unknown"
    cross_context: bool = False


@dataclass(frozen=True)
class MemoryContext:
    db1_results: list[EpisodicMemoryRecord] = field(default_factory=list)
    db2_results: list[SemanticMemoryRecord] = field(default_factory=list)

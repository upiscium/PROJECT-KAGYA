"""Versioned, minimal, durable AgentState snapshot authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from kagya.body import EmotionEngineAllostasis, EmotionState, EmotionTemporalState
from kagya.cognition.surprisal_calculator import (
    CalibrationEntry,
    LossCalibration,
)
from kagya.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
)
from kagya.identity.value_system import (
    ValueConflictDefinition,
    ValueDomainError,
    ValueRevisionHistory,
    ValueRevisionOperation,
    ValueRevisionRecord,
    ValueSeedDeclaration,
    ValueScope,
    ValueState,
    ValueSystem,
    recompute_seed_contract_digest,
)
from kagya.privacy import normalize_private_key
from kagya.runtime.context import (
    ContextFrame,
    ContextRegistry,
    ContextRegistryState,
    ContextStatus,
    ContextType,
    InterlocutorBinding,
    validate_context_registry_state,
)
from kagya.runtime.working_memory import (
    WorkingMemory,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySourceKind,
    working_memory_item_id,
)

if TYPE_CHECKING:
    from kagya.runtime.main_loop import KagyaMainLoop


CURRENT_AGENT_STATE_SCHEMA_VERSION: Literal[5] = 5


class _StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EmotionStateSnapshot(_StateModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    optimal_loss: float = Field(ge=0.0)

    @field_validator("valence", "arousal", "optimal_loss")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("emotion value must be finite")
        return value


class _AgentStateSnapshotBase(_StateModel):
    saved_at: datetime
    last_processed_event_sequence: int = Field(ge=0)
    emotion_state: EmotionStateSnapshot

    @field_validator("saved_at", mode="before")
    @classmethod
    def parse_saved_at(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("saved_at must be a valid datetime") from error
        return value

    @field_validator("saved_at")
    @classmethod
    def require_aware_saved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("saved_at must be timezone-aware")
        return value

    @field_validator("last_processed_event_sequence", mode="before")
    @classmethod
    def reject_boolean_sequence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("sequence must be an integer")
        return value


class AgentStateSnapshotV1(_AgentStateSnapshotBase):
    """Exact retained R04-R07 canonical AgentState schema."""

    schema_version: Literal[1] = 1


class WorkingMemoryItemSnapshot(_StateModel):
    """Strict durable form of one U1 authoritative reference item."""

    item_id: str
    source_kind: Literal["episodic", "semantic"]
    source_id: str
    activation: float = Field(ge=0.0, le=1.0)
    salience: float = Field(ge=0.0, le=1.0)
    retention_reason: Literal["recent", "reactivated"]
    created_revision: int = Field(ge=0)
    last_activated_revision: int = Field(ge=0)

    @field_validator("activation", "salience")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Working Memory value must be finite")
        return value

    @field_validator("created_revision", "last_activated_revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Working Memory revision must be an integer")
        return value

    @model_validator(mode="after")
    def require_canonical_reference(self) -> WorkingMemoryItemSnapshot:
        try:
            source_kind = WorkingMemorySourceKind(self.source_kind)
            expected = working_memory_item_id(source_kind, self.source_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Working Memory source reference is invalid") from error
        if self.item_id != expected:
            raise ValueError("Working Memory item identity is invalid")
        return self


class WorkingMemorySnapshot(_StateModel):
    """Canonical Working Memory authority embedded in AgentState v2."""

    revision: int = Field(ge=0)
    items: tuple[WorkingMemoryItemSnapshot, ...]

    @field_validator("items", mode="before")
    @classmethod
    def parse_json_items(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Working Memory revision must be an integer")
        return value

    @model_validator(mode="after")
    def require_consistent_membership(self) -> WorkingMemorySnapshot:
        item_ids: set[str] = set()
        source_references: set[tuple[str, str]] = set()
        for item in self.items:
            if (
                item.created_revision > self.revision
                or item.last_activated_revision > self.revision
            ):
                raise ValueError("Working Memory item revision is invalid")
            source_reference = (item.source_kind, item.source_id)
            if item.item_id in item_ids or source_reference in source_references:
                raise ValueError("Working Memory item is duplicated")
            item_ids.add(item.item_id)
            source_references.add(source_reference)
        return self


class AgentStateSnapshotV2(_AgentStateSnapshotBase):
    """Exact retained R08 canonical AgentState schema."""

    schema_version: Literal[2] = 2
    working_memory: WorkingMemorySnapshot


def _parse_context_timestamp(value: object) -> object:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Context timestamp must be a valid datetime") from error
    return value


def _require_context_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Context timestamp must be canonical UTC")
    return value.astimezone(timezone.utc)


def _tuple_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


class ContextFrameSnapshot(_StateModel):
    """Strict durable projection of one Context frame."""

    context_id: str
    context_type: Literal["conversation"]
    source_channel: str
    source_session_id: str | None
    participant_refs: tuple[str, ...]
    parent_context_id: str | None
    related_context_ids: tuple[str, ...]
    status: Literal["active", "suspended", "closed"]
    created_revision: int = Field(ge=0)
    last_modified_revision: int = Field(ge=0)
    started_at: datetime
    last_active_at: datetime

    @field_validator(
        "participant_refs", "related_context_ids", mode="before"
    )
    @classmethod
    def parse_reference_lists(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("created_revision", "last_modified_revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Context revision must be an integer")
        return value

    @field_validator("started_at", "last_active_at", mode="before")
    @classmethod
    def parse_timestamps(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("started_at", "last_active_at")
    @classmethod
    def require_utc_timestamps(cls, value: datetime) -> datetime:
        return _require_context_utc(value)


class InterlocutorBindingSnapshot(_StateModel):
    """Strict durable projection of one evidence-bound interlocutor binding."""

    reference_key: str
    identity_key: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_references: tuple[str, ...]
    created_revision: int = Field(ge=0)
    last_modified_revision: int = Field(ge=0)

    @field_validator("evidence_references", mode="before")
    @classmethod
    def parse_reference_list(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Context confidence must be finite")
        return value

    @field_validator("created_revision", "last_modified_revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Context revision must be an integer")
        return value


class ContextStateSnapshot(_StateModel):
    """Canonical Context authority embedded in AgentState v3."""

    revision: int = Field(ge=0)
    current_context_id: str | None
    frames: tuple[ContextFrameSnapshot, ...]
    interlocutor_bindings: tuple[InterlocutorBindingSnapshot, ...]

    @field_validator("frames", "interlocutor_bindings", mode="before")
    @classmethod
    def parse_snapshot_lists(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Context revision must be an integer")
        return value

    def to_registry_state(self) -> ContextRegistryState:
        return ContextRegistryState(
            revision=self.revision,
            current_context_id=self.current_context_id,
            frames=tuple(
                ContextFrame(
                    context_id=frame.context_id,
                    context_type=ContextType(frame.context_type),
                    source_channel=frame.source_channel,
                    source_session_id=frame.source_session_id,
                    participant_refs=frame.participant_refs,
                    parent_context_id=frame.parent_context_id,
                    related_context_ids=frame.related_context_ids,
                    status=ContextStatus(frame.status),
                    created_revision=frame.created_revision,
                    last_modified_revision=frame.last_modified_revision,
                    started_at=frame.started_at,
                    last_active_at=frame.last_active_at,
                )
                for frame in self.frames
            ),
            interlocutor_bindings=tuple(
                InterlocutorBinding(
                    reference_key=binding.reference_key,
                    identity_key=binding.identity_key,
                    confidence=binding.confidence,
                    evidence_references=binding.evidence_references,
                    created_revision=binding.created_revision,
                    last_modified_revision=binding.last_modified_revision,
                )
                for binding in self.interlocutor_bindings
            ),
        )

    @model_validator(mode="after")
    def require_valid_registry_state(self) -> ContextStateSnapshot:
        validate_context_registry_state(self.to_registry_state())
        return self


class AgentStateSnapshotV3(_AgentStateSnapshotBase):
    """Exact retained R09 v3 canonical snapshot."""

    schema_version: Literal[3] = 3
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


class CalibrationEntrySnapshot(_StateModel):
    """Opaque, exact persisted Welford calibration state."""

    model_key: str
    count: int = Field(ge=0)
    mean: float
    m2: float = Field(ge=0.0)

    @field_validator("model_key")
    @classmethod
    def require_opaque_key(cls, value: str) -> str:
        if len(value) != 70 or not value.startswith("model.") or any(
            character not in "0123456789abcdef" for character in value[6:]
        ):
            raise ValueError("model_key must be an opaque model key")
        return value

    @field_validator("count", mode="before")
    @classmethod
    def reject_boolean_count(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("count must be an integer")
        return value

    @field_validator("mean", "m2")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("calibration values must be finite")
        return value

    @model_validator(mode="after")
    def require_exact_statistics(self) -> CalibrationEntrySnapshot:
        # Use the authoritative U1 validator rather than duplicating its rules.
        CalibrationEntry(self.model_key, self.count, self.mean, self.m2)
        return self

    def to_entry(self) -> CalibrationEntry:
        return CalibrationEntry(self.model_key, self.count, self.mean, self.m2)


class AppraisalStateSnapshot(_StateModel):
    calibration_entries: tuple[CalibrationEntrySnapshot, ...]
    last_emotion_update_at: datetime | None

    @field_validator("calibration_entries", mode="before")
    @classmethod
    def parse_entries(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("last_emotion_update_at", mode="before")
    @classmethod
    def parse_timestamp(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("last_emotion_update_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_context_utc(value)

    @model_validator(mode="after")
    def require_ordered_unique_entries(self) -> AppraisalStateSnapshot:
        keys = tuple(entry.model_key for entry in self.calibration_entries)
        if len(keys) > 64:
            raise ValueError("calibration entries are bounded")
        if keys != tuple(sorted(set(keys))):
            raise ValueError("calibration entries must be ordered and unique")
        return self


class IdentityOriginSnapshot(_StateModel):
    """Strict durable projection of one Value provenance record."""

    actor: Literal[
        "self",
        "user",
        "operator",
        "system",
        "external_source",
        "model_inference",
        "inherited",
        "unknown",
    ]
    input_kind: Literal[
        "internal_state",
        "request",
        "suggestion",
        "constraint",
        "feedback",
        "evidence",
        "config_seed",
        "legacy",
    ]
    admission: Literal[
        "pending",
        "self_endorsed",
        "system_authorized",
        "rejected",
        "uncertain",
    ]
    source_ref: str | None
    event_id: str | None
    context_id: str | None
    event_sequence: int | None
    confidence: float

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Value origin confidence must be finite and bounded")
        return value

    @field_validator("event_sequence", mode="before")
    @classmethod
    def reject_boolean_event_sequence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value origin event_sequence must be an integer")
        return value


class ValueStateSnapshot(_StateModel):
    """Strict durable projection of one complete current Value state."""

    value_id: str
    revision: int = Field(ge=0)
    name: str
    concept: str | None
    scope: Literal["subject", "context"]
    context_ids: tuple[str, ...]
    polarity: int
    strength: float
    confidence: float
    stability: float
    protectedness: float
    negotiability: float
    allowed_update_rate: float
    frozen: bool
    origin: IdentityOriginSnapshot
    evidence_refs: tuple[str, ...]
    seed_contract_digest: str | None
    opposition_count: int = Field(ge=0, le=6)

    @field_validator("context_ids", "evidence_refs", mode="before")
    @classmethod
    def parse_value_lists(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator(
        "strength",
        "confidence",
        "stability",
        "protectedness",
        "negotiability",
        "allowed_update_rate",
    )
    @classmethod
    def require_finite_value(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Value scalar must be finite and bounded")
        return value

    @field_validator("revision", "opposition_count", mode="before")
    @classmethod
    def reject_boolean_value_integer(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value integer fields must be integers")
        return value


class ValueConflictSnapshot(_StateModel):
    left_value_id: str
    right_value_id: str


class ValueRevisionRecordSnapshot(_StateModel):
    """Strict durable projection of one immutable Value revision record."""

    value_id: str
    from_revision: int
    to_revision: int
    before_digest: str
    after_state_projection: ValueStateSnapshot
    after_digest: str
    operation: Literal["admission", "update", "freeze", "unfreeze", "rollback"]
    origin_id: str
    evidence_refs: tuple[str, ...]
    event_id: str
    event_sequence: int
    recorded_at: datetime
    previous_record_digest: str | None
    target_revision: int | None
    record_digest: str

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def parse_record_refs(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("recorded_at", mode="before")
    @classmethod
    def parse_record_timestamp(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("recorded_at")
    @classmethod
    def require_record_utc(cls, value: datetime) -> datetime:
        return _require_context_utc(value)

    @field_validator(
        "from_revision", "to_revision", "event_sequence", "target_revision", mode="before"
    )
    @classmethod
    def reject_boolean_record_integer(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value revision fields must be integers")
        return value


class ValueRevisionHistorySnapshot(_StateModel):
    value_id: str
    history_anchor_revision: int | None
    history_anchor_digest: str | None
    history_anchor_state_digest: str | None
    records: tuple[ValueRevisionRecordSnapshot, ...]

    @field_validator("records", mode="before")
    @classmethod
    def parse_history_records(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("history_anchor_revision", mode="before")
    @classmethod
    def reject_boolean_anchor_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("history_anchor_revision must be an integer")
        return value


class ValueEvidenceLedgerSnapshot(_StateModel):
    value_id: str
    evidence_refs: tuple[str, ...]
    ledger_digest: str

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def parse_ledger_refs(cls, value: object) -> object:
        return _tuple_value(value)


class ValueSystemStateSnapshot(_StateModel):
    """Complete durable Value authority embedded in AgentState v5."""

    schema_version: Literal[1] = 1
    values: tuple[ValueStateSnapshot, ...]
    conflicts: tuple[ValueConflictSnapshot, ...]
    histories: tuple[ValueRevisionHistorySnapshot, ...]
    evidence_ledgers: tuple[ValueEvidenceLedgerSnapshot, ...]

    @field_validator("values", "conflicts", "histories", "evidence_ledgers", mode="before")
    @classmethod
    def parse_value_snapshot_lists(cls, value: object) -> object:
        return _tuple_value(value)

class AgentStateSnapshotV4(_AgentStateSnapshotBase):
    """Exact retained R10 canonical AgentState v4 schema."""

    schema_version: Literal[4] = 4
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot
    appraisal_state: AppraisalStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


class AgentStateSnapshotV5(_AgentStateSnapshotBase):
    """Current AgentState v5 with complete Value authority continuity."""

    schema_version: Literal[5] = CURRENT_AGENT_STATE_SCHEMA_VERSION
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot
    appraisal_state: AppraisalStateSnapshot
    value_state: ValueSystemStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


# The unqualified name denotes the current schema; retained callers should use
# AgentStateSnapshotV4 when they intentionally construct the exact v4 shape.
AgentStateSnapshot = AgentStateSnapshotV5


CompatibleAgentStateSnapshot = Annotated[
    AgentStateSnapshotV1
    | AgentStateSnapshotV2
    | AgentStateSnapshotV3
    | AgentStateSnapshotV4
    | AgentStateSnapshotV5,
    Field(discriminator="schema_version"),
]
_COMPATIBLE_SNAPSHOT_ADAPTER: TypeAdapter[CompatibleAgentStateSnapshot] = (
    TypeAdapter(CompatibleAgentStateSnapshot)
)


def validate_compatible_agent_state_snapshot(
    value: object,
) -> CompatibleAgentStateSnapshot:
    """Validate a compatible snapshot without changing its schema version."""

    return _COMPATIBLE_SNAPSHOT_ADAPTER.validate_python(value)


class _LegacyEmotionState(_StateModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    optimal_loss: float = Field(ge=0.0)

    @field_validator("valence", "arousal", "optimal_loss")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("emotion value must be finite")
        return value


class _LegacyAgentStateV0(_StateModel):
    schema_version: Literal[0]
    last_event_sequence: int = Field(ge=0)
    emotion: _LegacyEmotionState

    @field_validator("last_event_sequence", mode="before")
    @classmethod
    def reject_boolean_sequence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("sequence must be an integer")
        return value


class AgentStateError(Exception):
    """Base class for bounded AgentState failures."""


class AgentStateLoadError(AgentStateError):
    """The canonical snapshot exists but cannot be loaded safely."""


class AgentStateConfigurationDrift(AgentStateLoadError):
    """Persisted Value seed/configuration evidence no longer matches settings."""


class UnsupportedAgentStateVersion(AgentStateLoadError):
    """The canonical snapshot uses an unsupported schema version."""


class AgentStateSaveStage(str, Enum):
    CAPTURE = "snapshot_capture"
    TEMP_WRITE = "snapshot_temp_write"
    TEMP_FSYNC = "snapshot_temp_fsync"
    ATOMIC_REPLACE = "snapshot_atomic_replace"
    PARENT_FSYNC = "snapshot_parent_fsync"


class AgentStateSaveError(AgentStateError):
    """A snapshot did not reach confirmed durable success."""

    def __init__(self, stage: AgentStateSaveStage, *, published: bool) -> None:
        self.stage = stage
        self.published = published
        super().__init__(
            "AgentState snapshot save failed "
            f"at {stage.value}; published={str(published).lower()}"
        )


_PRIVATE_STATE_KEYS = frozenset(
    {
        "hiddenthought",
        "privatereasoning",
        "reasoning",
        "chainofthought",
        "thought",
        "content",
        "prompt",
        "rawprompt",
        "systemprompt",
        "userprompt",
        "userinput",
        "assistantprompt",
        "response",
        "retrievedmemory",
        "privatestate",
        "turns",
        "turn",
        "sessionturns",
        "attachments",
        "attachment",
        "eventpayload",
        "requestpayload",
        "debugtrace",
        "debugchattrace",
        "chattranscript",
        "transcript",
    }
)


def _reject_private_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if normalize_private_key(key) in _PRIVATE_STATE_KEYS:
                raise ValueError("snapshot contains a forbidden field")
            _reject_private_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_private_keys(child)


def _origin_snapshot(origin: IdentityOrigin) -> IdentityOriginSnapshot:
    return IdentityOriginSnapshot(
        actor=origin.actor.value,
        input_kind=origin.input_kind.value,
        admission=origin.admission.value,
        source_ref=origin.source_ref,
        event_id=origin.event_id,
        context_id=origin.context_id,
        event_sequence=origin.event_sequence,
        confidence=origin.confidence,
    )


def _value_snapshot(value: ValueState) -> ValueStateSnapshot:
    return ValueStateSnapshot(
        value_id=value.value_id,
        revision=value.revision,
        name=value.name,
        concept=value.concept,
        scope=value.scope.value,
        context_ids=value.context_ids,
        polarity=value.polarity,
        strength=value.strength,
        confidence=value.confidence,
        stability=value.stability,
        protectedness=value.protectedness,
        negotiability=value.negotiability,
        allowed_update_rate=value.allowed_update_rate,
        frozen=value.frozen,
        origin=_origin_snapshot(value.origin),
        evidence_refs=value.evidence_refs,
        seed_contract_digest=value.seed_contract_digest,
        opposition_count=value.opposition_count,
    )


def _revision_record_snapshot(record: ValueRevisionRecord) -> ValueRevisionRecordSnapshot:
    return ValueRevisionRecordSnapshot(
        value_id=record.value_id,
        from_revision=record.from_revision,
        to_revision=record.to_revision,
        before_digest=record.before_digest,
        after_state_projection=_value_snapshot(record.after_state_projection),
        after_digest=record.after_digest,
        operation=record.operation.value,
        origin_id=record.origin_id,
        evidence_refs=record.evidence_refs,
        event_id=record.event_id,
        event_sequence=record.event_sequence,
        recorded_at=record.recorded_at,
        previous_record_digest=record.previous_record_digest,
        target_revision=record.target_revision,
        record_digest=record.record_digest,
    )


def _history_snapshot(history: ValueRevisionHistory) -> ValueRevisionHistorySnapshot:
    return ValueRevisionHistorySnapshot(
        value_id=history.value_id,
        history_anchor_revision=history.history_anchor_revision,
        history_anchor_digest=history.history_anchor_digest,
        history_anchor_state_digest=history.history_anchor_state_digest,
        records=tuple(_revision_record_snapshot(record) for record in history.records),
    )


def _value_state_snapshot(system: ValueSystem) -> ValueSystemStateSnapshot:
    snapshot = system.snapshot()
    digests = dict(snapshot.evidence_ledger_digests)
    return ValueSystemStateSnapshot(
        schema_version=1,
        values=tuple(_value_snapshot(value) for value in snapshot.values),
        conflicts=tuple(
            ValueConflictSnapshot(
                left_value_id=conflict.left_value_id,
                right_value_id=conflict.right_value_id,
            )
            for conflict in snapshot.conflicts
        ),
        histories=tuple(_history_snapshot(history) for history in snapshot.histories),
        evidence_ledgers=tuple(
            ValueEvidenceLedgerSnapshot(
                value_id=value_id,
                evidence_refs=ledger,
                ledger_digest=digests[value_id],
            )
            for value_id, ledger in snapshot.evidence_ledgers
        ),
    )


def _identity_origin(snapshot: IdentityOriginSnapshot) -> IdentityOrigin:
    return IdentityOrigin(
        OriginActor(snapshot.actor),
        OriginInputKind(snapshot.input_kind),
        ValueAdmissionStatus(snapshot.admission),
        source_ref=snapshot.source_ref,
        event_id=snapshot.event_id,
        context_id=snapshot.context_id,
        event_sequence=snapshot.event_sequence,
        confidence=snapshot.confidence,
    )


def _domain_value(snapshot: ValueStateSnapshot) -> ValueState:
    return ValueState(
        value_id=snapshot.value_id,
        revision=snapshot.revision,
        name=snapshot.name,
        concept=snapshot.concept,
        scope=ValueScope(snapshot.scope),
        context_ids=snapshot.context_ids,
        polarity=snapshot.polarity,
        strength=snapshot.strength,
        confidence=snapshot.confidence,
        stability=snapshot.stability,
        protectedness=snapshot.protectedness,
        negotiability=snapshot.negotiability,
        allowed_update_rate=snapshot.allowed_update_rate,
        frozen=snapshot.frozen,
        origin=_identity_origin(snapshot.origin),
        evidence_refs=snapshot.evidence_refs,
        seed_contract_digest=snapshot.seed_contract_digest,
        opposition_count=snapshot.opposition_count,
    )


def _domain_record(snapshot: ValueRevisionRecordSnapshot) -> ValueRevisionRecord:
    record = ValueRevisionRecord(
        value_id=snapshot.value_id,
        from_revision=snapshot.from_revision,
        to_revision=snapshot.to_revision,
        before_digest=snapshot.before_digest,
        after_state_projection=_domain_value(snapshot.after_state_projection),
        after_digest=snapshot.after_digest,
        operation=ValueRevisionOperation(snapshot.operation),
        origin_id=snapshot.origin_id,
        evidence_refs=snapshot.evidence_refs,
        event_id=snapshot.event_id,
        event_sequence=snapshot.event_sequence,
        recorded_at=snapshot.recorded_at,
        previous_record_digest=snapshot.previous_record_digest,
        target_revision=snapshot.target_revision,
    )
    if record.record_digest != snapshot.record_digest:
        raise ValueDomainError("revision record digest does not match persisted record")
    return record


def _domain_history(snapshot: ValueRevisionHistorySnapshot) -> ValueRevisionHistory:
    return ValueRevisionHistory(
        value_id=snapshot.value_id,
        history_anchor_revision=snapshot.history_anchor_revision,
        history_anchor_digest=snapshot.history_anchor_digest,
        history_anchor_state_digest=snapshot.history_anchor_state_digest,
        records=tuple(_domain_record(record) for record in snapshot.records),
    )


def _domain_value_system(snapshot: ValueSystemStateSnapshot) -> ValueSystem:
    try:
        values = {_value.value_id: _domain_value(_value) for _value in snapshot.values}
        histories = {
            _history.value_id: _domain_history(_history)
            for _history in snapshot.histories
        }
        ledgers = {
            ledger.value_id: ledger.evidence_refs
            for ledger in snapshot.evidence_ledgers
        }
        ledger_digests = {
            ledger.value_id: ledger.ledger_digest
            for ledger in snapshot.evidence_ledgers
        }
        if len(values) != len(snapshot.values):
            raise ValueDomainError("Value snapshot contains duplicate Value IDs")
        if len(histories) != len(snapshot.histories):
            raise ValueDomainError("Value snapshot contains duplicate history IDs")
        if len(ledgers) != len(snapshot.evidence_ledgers):
            raise ValueDomainError("Value snapshot contains duplicate ledger IDs")
        conflicts = tuple(
            ValueConflictDefinition(
                left_value_id=conflict.left_value_id,
                right_value_id=conflict.right_value_id,
            )
            for conflict in snapshot.conflicts
        )
        return ValueSystem.restore(
            values=values,
            conflicts=conflicts,
            histories=histories,
            evidence_ledgers=ledgers,
            evidence_ledger_digests=ledger_digests,
        )
    except Exception:
        raise AgentStateLoadError("ValueSystem snapshot is invalid") from None


def default_agent_state_snapshot(
    baseline_surprisal: float,
    *,
    saved_at: datetime | None = None,
    value_system: ValueSystem | None = None,
) -> AgentStateSnapshotV5:
    """Return the bootstrap state used only when the canonical file is absent."""

    system = value_system if value_system is not None else ValueSystem()
    return AgentStateSnapshotV5(
        saved_at=saved_at or datetime.now(timezone.utc),
        last_processed_event_sequence=0,
        emotion_state=EmotionStateSnapshot(
            valence=0.0,
            arousal=0.0,
            optimal_loss=baseline_surprisal,
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
        context_state=ContextStateSnapshot(
            revision=0,
            current_context_id=None,
            frames=(),
            interlocutor_bindings=(),
        ),
        appraisal_state=AppraisalStateSnapshot(
            calibration_entries=(), last_emotion_update_at=None
        ),
        value_state=_value_state_snapshot(system),
    )


class AgentStateStore:
    """Load, capture, restore, and atomically publish the R04 snapshot."""

    def __init__(
        self,
        path: str | Path,
        baseline_surprisal: float,
        *,
        value_seeds: tuple[ValueSeedDeclaration, ...] = (),
        value_conflicts: tuple[ValueConflictDefinition, ...] = (),
        clock: Callable[[], datetime] | None = None,
        save_stage_hook: Callable[[AgentStateSaveStage], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._baseline_surprisal = baseline_surprisal
        self._value_seeds: tuple[ValueSeedDeclaration, ...] = ()
        self._value_conflicts: tuple[ValueConflictDefinition, ...] = ()
        self._configured_value_system = ValueSystem()
        self.configure_value_contract(value_seeds, value_conflicts)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._save_stage_hook = save_stage_hook

    @property
    def configured_value_system(self) -> ValueSystem:
        """Return a fresh configured baseline for legacy/fresh startup."""

        return ValueSystem.restore_snapshot(self._configured_value_system.snapshot())

    def configure_value_contract(
        self,
        value_seeds: tuple[ValueSeedDeclaration, ...],
        value_conflicts: tuple[ValueConflictDefinition, ...],
    ) -> None:
        """Bind immutable configuration evidence before loading durable state."""

        seeds = tuple(value_seeds)
        conflicts = tuple(value_conflicts)
        if any(not isinstance(seed, ValueSeedDeclaration) for seed in seeds):
            raise TypeError("value_seeds must contain ValueSeedDeclaration values")
        if any(not isinstance(conflict, ValueConflictDefinition) for conflict in conflicts):
            raise TypeError(
                "value_conflicts must contain ValueConflictDefinition values"
            )
        conflicts = tuple(
            sorted(
                conflicts,
                key=lambda conflict: (
                    conflict.left_value_id,
                    conflict.right_value_id,
                ),
            )
        )
        configured = ValueSystem.from_seed_declarations(seeds, conflicts)
        self._value_seeds = seeds
        self._value_conflicts = conflicts
        self._configured_value_system = configured

    def snapshot_exists(self) -> bool:
        """Inspect canonical snapshot presence without following its final path."""

        try:
            self.path.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            raise AgentStateLoadError(
                "AgentState snapshot cannot be inspected"
            ) from None
        return True

    def load(self) -> CompatibleAgentStateSnapshot:
        inspection_failure: AgentStateLoadError | None = None
        try:
            path_status = self.path.lstat()
        except FileNotFoundError:
            try:
                return default_agent_state_snapshot(
                    self._baseline_surprisal,
                    saved_at=self._now(),
                    value_system=self.configured_value_system,
                )
            except Exception:
                inspection_failure = AgentStateLoadError("AgentState bootstrap failed")
        except OSError:
            inspection_failure = AgentStateLoadError(
                "AgentState snapshot cannot be inspected"
            )
        if inspection_failure is not None:
            raise inspection_failure

        if not stat.S_ISREG(path_status.st_mode):
            raise AgentStateLoadError("AgentState snapshot is not a regular file")

        read_failure: AgentStateLoadError | None = None
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("snapshot is not a regular file")
                with os.fdopen(descriptor, "rb") as snapshot_file:
                    descriptor = -1
                    raw_bytes = snapshot_file.read()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            raw = json.loads(
                raw_bytes.decode("utf-8"),
                parse_constant=self._reject_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            read_failure = AgentStateLoadError("AgentState snapshot is malformed")
        if read_failure is not None:
            raise read_failure

        if not isinstance(raw, dict):
            raise AgentStateLoadError("AgentState snapshot root is invalid")
        privacy_failure: AgentStateLoadError | None = None
        try:
            _reject_private_keys(raw)
        except ValueError:
            privacy_failure = AgentStateLoadError(
                "AgentState snapshot violates privacy"
            )
        if privacy_failure is not None:
            raise privacy_failure

        version = raw.get("schema_version")
        if version == 1:
            schema_failure: AgentStateLoadError | None = None
            try:
                return AgentStateSnapshotV1.model_validate(raw)
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == 2:
            schema_failure = None
            try:
                return AgentStateSnapshotV2.model_validate(raw)
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == 3:
            try:
                return AgentStateSnapshotV3.model_validate(raw)
            except ValidationError:
                raise AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                ) from None
        if version == 4:
            try:
                return AgentStateSnapshotV4.model_validate(raw)
            except ValidationError:
                raise AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                ) from None
        if version == CURRENT_AGENT_STATE_SCHEMA_VERSION:
            schema_failure = None
            try:
                loaded = AgentStateSnapshotV5.model_validate(raw)
                self._validate_value_configuration(loaded)
                _domain_value_system(loaded.value_state)
                return loaded
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            except AgentStateLoadError:
                raise
            except Exception:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == 0:
            return self._migrate_v0(raw)
        if isinstance(version, int) and not isinstance(version, bool):
            raise UnsupportedAgentStateVersion(
                "AgentState schema version is unsupported"
            )
        raise AgentStateLoadError("AgentState schema version is invalid")

    def _validate_value_configuration(self, snapshot: AgentStateSnapshotV5) -> None:
        """Check configuration as compatibility evidence, never as overwrite authority."""

        if not self._value_seeds and not self._value_conflicts:
            return
        persisted = {value.value_id: value for value in snapshot.value_state.values}
        for seed in self._value_seeds:
            value = persisted.get(seed.value_id)
            if value is None:
                # A newly declared seed is not adopted into an existing snapshot.
                continue
            if value.origin.admission != ValueAdmissionStatus.SYSTEM_AUTHORIZED.value:
                raise AgentStateConfigurationDrift(
                    "Value configuration collides with persisted non-system state"
                )
            if value.seed_contract_digest != recompute_seed_contract_digest(seed):
                raise AgentStateConfigurationDrift(
                    "Persisted system Value seed declaration has drifted"
                )

    def save(self, snapshot: CompatibleAgentStateSnapshot) -> None:
        stage = AgentStateSaveStage.TEMP_WRITE
        published = False
        temporary_path: Path | None = None
        descriptor: int | None = None
        save_failure: AgentStateSaveError | None = None
        try:
            payload = self.canonical_bytes(snapshot)
            parent = self.path.parent
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=parent
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as temporary_file:
                descriptor = None
                self._run_stage(stage)
                if temporary_file.write(payload) != len(payload):
                    raise OSError("incomplete snapshot write")
                temporary_file.flush()
                stage = AgentStateSaveStage.TEMP_FSYNC
                self._run_stage(stage)
                os.fsync(temporary_file.fileno())

            stage = AgentStateSaveStage.ATOMIC_REPLACE
            self._run_stage(stage)
            os.replace(temporary_path, self.path)
            published = True

            stage = AgentStateSaveStage.PARENT_FSYNC
            self._run_stage(stage)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except Exception:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path is not None and not published:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
            save_failure = AgentStateSaveError(stage, published=published)
        if save_failure is not None:
            raise save_failure

    def canonical_bytes(self, snapshot: CompatibleAgentStateSnapshot) -> bytes:
        """Return the one canonical representation used for save and hashing."""

        canonical_failure: AgentStateSaveError | None = None
        try:
            raw: object = snapshot.model_dump(mode="python")
            _reject_private_keys(raw)
            validated = validate_compatible_agent_state_snapshot(raw)
            if isinstance(validated, AgentStateSnapshotV5):
                self._validate_value_configuration(validated)
                _domain_value_system(validated.value_state)
            return self._canonical_bytes(validated)
        except (AgentStateConfigurationDrift, AgentStateLoadError):
            raise
        except Exception:
            canonical_failure = AgentStateSaveError(
                AgentStateSaveStage.CAPTURE, published=False
            )
        raise canonical_failure

    def snapshot_hash(self, snapshot: CompatibleAgentStateSnapshot) -> str:
        """Hash the exact canonical bytes published by this store."""

        return hashlib.sha256(self.canonical_bytes(snapshot)).hexdigest()

    def ensure_published(self, snapshot: CompatibleAgentStateSnapshot) -> None:
        """Publish bootstrap/migrated state while avoiding an identical rewrite."""

        try:
            preserve_legacy = (
                isinstance(
                    snapshot,
                    (
                        AgentStateSnapshotV1,
                        AgentStateSnapshotV2,
                        AgentStateSnapshotV3,
                        AgentStateSnapshotV4,
                    ),
                )
                and self.snapshot_exists()
            )
        except AgentStateLoadError:
            preserve_legacy = False
        if preserve_legacy:
            try:
                published_snapshot = self.load()
            except AgentStateLoadError:
                pass
            else:
                if published_snapshot == snapshot:
                    return
        payload = self.canonical_bytes(snapshot)
        inspection_failure: AgentStateSaveError | None = None
        descriptor: int | None = None
        try:
            status = self.path.lstat()
        except FileNotFoundError:
            self.save(snapshot)
            return
        except OSError:
            inspection_failure = AgentStateSaveError(
                AgentStateSaveStage.TEMP_WRITE, published=False
            )
        else:
            try:
                if not stat.S_ISREG(status.st_mode):
                    raise OSError("snapshot is not a regular file")
                descriptor = os.open(
                    self.path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("snapshot is not a regular file")
                with os.fdopen(descriptor, "rb") as snapshot_file:
                    descriptor = None
                    published = snapshot_file.read()
            except OSError:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                inspection_failure = AgentStateSaveError(
                    AgentStateSaveStage.TEMP_WRITE, published=False
                )
            else:
                if published == payload:
                    return
        if inspection_failure is not None:
            raise inspection_failure
        self.save(snapshot)

    def capture(
        self, main_loop: KagyaMainLoop, sequence: int
    ) -> AgentStateSnapshotV5:
        capture_failure: AgentStateSaveError | None = None
        try:
            emotion_engine = getattr(main_loop, "emotion_engine", None)
            if not isinstance(emotion_engine, EmotionEngineAllostasis):
                raise ValueError("Emotion authority is unavailable")
            working_memory = getattr(main_loop, "working_memory", None)
            if not isinstance(working_memory, WorkingMemory):
                raise ValueError("Working Memory authority is unavailable")
            context_registry = getattr(main_loop, "context_registry", None)
            if not isinstance(context_registry, ContextRegistry):
                raise ValueError("Context authority is unavailable")
            emotion = emotion_engine.state
            context_state = context_registry.state
            validate_context_registry_state(context_state)
            calibration = getattr(main_loop, "loss_calibration")
            if not isinstance(calibration, LossCalibration):
                raise ValueError("Calibration authority is unavailable")
            exported_calibration = calibration.export()
            if not isinstance(exported_calibration, tuple):
                raise ValueError("Calibration authority is malformed")
            calibration_entries = tuple(
                CalibrationEntrySnapshot(
                    model_key=entry.model_key,
                    count=entry.count,
                    mean=entry.mean,
                    m2=entry.m2,
                )
                for entry in exported_calibration
            )
            temporal = getattr(emotion_engine, "temporal_state")
            if not isinstance(temporal, EmotionTemporalState):
                raise ValueError("Emotion temporal authority is malformed")
            value_system = getattr(main_loop, "value_system", None)
            if not isinstance(value_system, ValueSystem):
                raise ValueError("Value authority is unavailable")
            common: dict[str, Any] = dict(
                saved_at=self._now(),
                last_processed_event_sequence=sequence,
                emotion_state=EmotionStateSnapshot(
                    valence=emotion.valence,
                    arousal=emotion.arousal,
                    optimal_loss=emotion.optimal_loss,
                ),
                working_memory=WorkingMemorySnapshot(
                    revision=working_memory.revision,
                    items=tuple(
                        WorkingMemoryItemSnapshot(
                            item_id=item.item_id,
                            source_kind=item.source_kind.value,
                            source_id=item.source_id,
                            activation=item.activation,
                            salience=item.salience,
                            retention_reason=item.retention_reason.value,
                            created_revision=item.created_revision,
                            last_activated_revision=item.last_activated_revision,
                        )
                        for item in working_memory.items
                    ),
                ),
                context_state=ContextStateSnapshot(
                    revision=context_state.revision,
                    current_context_id=context_state.current_context_id,
                    frames=tuple(
                        ContextFrameSnapshot(
                            context_id=frame.context_id,
                            context_type=cast(
                                Literal["conversation"], frame.context_type.value
                            ),
                            source_channel=frame.source_channel,
                            source_session_id=frame.source_session_id,
                            participant_refs=frame.participant_refs,
                            parent_context_id=frame.parent_context_id,
                            related_context_ids=frame.related_context_ids,
                            status=cast(
                                Literal["active", "suspended", "closed"],
                                frame.status.value,
                            ),
                            created_revision=frame.created_revision,
                            last_modified_revision=frame.last_modified_revision,
                            started_at=frame.started_at,
                            last_active_at=frame.last_active_at,
                        )
                        for frame in context_state.frames
                    ),
                    interlocutor_bindings=tuple(
                        InterlocutorBindingSnapshot(
                            reference_key=binding.reference_key,
                            identity_key=binding.identity_key,
                            confidence=binding.confidence,
                            evidence_references=binding.evidence_references,
                            created_revision=binding.created_revision,
                            last_modified_revision=binding.last_modified_revision,
                        )
                        for binding in context_state.interlocutor_bindings
                    ),
                ),
                appraisal_state=AppraisalStateSnapshot(
                    calibration_entries=calibration_entries,
                    last_emotion_update_at=temporal.last_update_at,
                ),
            )
            return AgentStateSnapshotV5(
                **common,
                value_state=_value_state_snapshot(value_system),
            )
        except Exception:
            capture_failure = AgentStateSaveError(
                AgentStateSaveStage.CAPTURE, published=False
            )
        raise capture_failure

    def restore_into(
        self, main_loop: KagyaMainLoop, snapshot: CompatibleAgentStateSnapshot
    ) -> None:
        restore_failure: AgentStateLoadError | None = None
        context_registry: ContextRegistry | None = None
        calibration: LossCalibration | None = None
        previous_emotion: EmotionState | None = None
        previous_working_memory_revision: int | None = None
        previous_working_memory_items: tuple[WorkingMemoryItem, ...] | None = None
        previous_context: ContextRegistryState | None = None
        previous_calibration: tuple[CalibrationEntry, ...] | None = None
        previous_temporal: EmotionTemporalState | None = None
        previous_value_system: ValueSystem | None = None
        value_system_authority: ValueSystem | None = None
        value_system_was_present = False
        emotion_engine: EmotionEngineAllostasis | None = None
        working_memory_authority: WorkingMemory | None = None
        try:
            validated = validate_compatible_agent_state_snapshot(
                snapshot.model_dump(mode="python")
            )
            if isinstance(validated, AgentStateSnapshotV5):
                self._validate_value_configuration(validated)
                restored_value_system = _domain_value_system(validated.value_state)
            else:
                restored_value_system = self.configured_value_system
            emotion_engine = getattr(main_loop, "emotion_engine", None)
            if not isinstance(emotion_engine, EmotionEngineAllostasis):
                raise AgentStateLoadError("AgentState restore requires EmotionEngine")
            working_memory_authority = getattr(main_loop, "working_memory", None)
            if not isinstance(working_memory_authority, WorkingMemory):
                raise AgentStateLoadError("AgentState restore requires WorkingMemory")
            emotion = validated.emotion_state
            working_memory = (
                validated.working_memory
                if isinstance(
                    validated,
                    (
                        AgentStateSnapshotV2,
                        AgentStateSnapshotV3,
                        AgentStateSnapshotV4,
                        AgentStateSnapshotV5,
                    ),
                )
                else WorkingMemorySnapshot(revision=0, items=())
            )
            context_registry = getattr(main_loop, "context_registry", None)
            if context_registry is not None and not isinstance(
                context_registry, ContextRegistry
            ):
                raise AgentStateLoadError("AgentState restore requires ContextRegistry")
            context_state = (
                validated.context_state.to_registry_state()
                if isinstance(
                    validated,
                    (AgentStateSnapshotV3, AgentStateSnapshotV4, AgentStateSnapshotV5),
                )
                else ContextRegistryState(0, None, (), ())
            )
            validate_context_registry_state(context_state)
            calibration = getattr(main_loop, "loss_calibration", None)
            if not isinstance(calibration, LossCalibration):
                raise AgentStateLoadError("AgentState restore requires LossCalibration")
            appraisal = (
                validated.appraisal_state
                if isinstance(validated, (AgentStateSnapshotV4, AgentStateSnapshotV5))
                else AppraisalStateSnapshot(
                    calibration_entries=(), last_emotion_update_at=None
                )
            )
            restored_calibration = tuple(
                entry.to_entry() for entry in appraisal.calibration_entries
            )
            restored_temporal = EmotionTemporalState(appraisal.last_emotion_update_at)
            current_temporal = getattr(emotion_engine, "temporal_state", None)
            if not isinstance(current_temporal, EmotionTemporalState):
                raise AgentStateLoadError(
                    "AgentState restore requires EmotionTemporalState"
                )
            restored_items = tuple(
                WorkingMemoryItem(
                    item_id=item.item_id,
                    source_kind=WorkingMemorySourceKind(item.source_kind),
                    source_id=item.source_id,
                    activation=item.activation,
                    salience=item.salience,
                    retention_reason=WorkingMemoryRetentionReason(
                        item.retention_reason
                    ),
                    created_revision=item.created_revision,
                    last_activated_revision=item.last_activated_revision,
                )
                for item in working_memory.items
            )
            if (
                isinstance(
                    validated,
                    (AgentStateSnapshotV3, AgentStateSnapshotV4, AgentStateSnapshotV5),
                )
                and context_registry is None
            ):
                raise AgentStateLoadError("AgentState restore requires ContextRegistry")

            current_value_system = getattr(main_loop, "value_system", None)
            if isinstance(validated, AgentStateSnapshotV5):
                if not isinstance(current_value_system, ValueSystem):
                    raise AgentStateLoadError(
                        "AgentState restore requires ValueSystem authority"
                    )
                value_system_authority = current_value_system
                previous_value_system = current_value_system
                value_system_was_present = True
            elif isinstance(current_value_system, ValueSystem):
                value_system_authority = current_value_system
                previous_value_system = current_value_system
                value_system_was_present = True

            previous_emotion = emotion_engine.state
            previous_working_memory_revision = working_memory_authority.revision
            previous_working_memory_items = working_memory_authority.items
            previous_context = (
                context_registry.state if context_registry is not None else None
            )
            previous_calibration = calibration.export()
            previous_temporal = current_temporal
            working_memory_authority.restore_exact(
                working_memory.revision,
                restored_items,
            )
            if context_registry is not None:
                context_registry.restore_exact(context_state)
            calibration.restore_exact(restored_calibration)
            if value_system_authority is not None:
                main_loop.value_system = restored_value_system
            emotion_engine.state = EmotionState(
                valence=emotion.valence,
                arousal=emotion.arousal,
                optimal_loss=emotion.optimal_loss,
            )
            emotion_engine.temporal_state = restored_temporal
        except (AgentStateConfigurationDrift, AgentStateLoadError):
            raise
        except Exception:
            if (
                previous_working_memory_revision is not None
                and previous_working_memory_items is not None
                and working_memory_authority is not None
            ):
                try:
                    working_memory_authority.restore_exact(
                        previous_working_memory_revision,
                        previous_working_memory_items,
                    )
                except Exception:
                    pass
            if previous_context is not None and context_registry is not None:
                try:
                    context_registry.restore_exact(previous_context)
                except Exception:
                    pass
            if previous_calibration is not None and calibration is not None:
                try:
                    calibration.restore_exact(previous_calibration)
                except Exception:
                    pass
            if previous_emotion is not None and emotion_engine is not None:
                try:
                    emotion_engine.state = previous_emotion
                except Exception:
                    pass
            if previous_temporal is not None and emotion_engine is not None:
                try:
                    emotion_engine.temporal_state = previous_temporal
                except Exception:
                    pass
            if value_system_was_present and previous_value_system is not None:
                try:
                    main_loop.value_system = previous_value_system
                except Exception:
                    pass
            restore_failure = AgentStateLoadError("AgentState restore failed")
        if restore_failure is not None:
            raise restore_failure

    def _migrate_v0(self, raw: dict[str, Any]) -> AgentStateSnapshotV5:
        migration_failure: AgentStateLoadError | None = None
        try:
            legacy = _LegacyAgentStateV0.model_validate(raw)
            return default_agent_state_snapshot(
                legacy.emotion.optimal_loss,
                saved_at=self._now(),
                value_system=self.configured_value_system,
            ).model_copy(
                update={
                    "last_processed_event_sequence": legacy.last_event_sequence,
                    "emotion_state": EmotionStateSnapshot(
                        valence=legacy.emotion.valence,
                        arousal=legacy.emotion.arousal,
                        optimal_loss=legacy.emotion.optimal_loss,
                    ),
                }
            )
        except Exception:
            migration_failure = AgentStateLoadError("AgentState v0 migration failed")
        raise migration_failure

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("AgentState clock must return timezone-aware UTC")
        return value.astimezone(timezone.utc)

    def _run_stage(self, stage: AgentStateSaveStage) -> None:
        if self._save_stage_hook is not None:
            self._save_stage_hook(stage)

    @staticmethod
    def _canonical_bytes(snapshot: CompatibleAgentStateSnapshot) -> bytes:
        if isinstance(snapshot, AgentStateSnapshotV5):
            _domain_value_system(snapshot.value_state)
        return json.dumps(
            snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _reject_json_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

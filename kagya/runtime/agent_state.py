"""Versioned durable state for the single KAGYA subject."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
import json
import os
import tempfile

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kagya.body import EmotionState
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemoryItem,
    WorkingMemoryKind,
)
from kagya.runtime.context import (
    ContextFrame,
    ContextStatus,
    InferredAttribute,
    InterlocutorModel,
)


CURRENT_AGENT_STATE_SCHEMA_VERSION = 3


class _StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmotionStateSnapshot(_StateModel):
    valence: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    arousal: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    optimal_loss: float = Field(ge=0.0, allow_inf_nan=False)


class WorkingMemoryItemSnapshot(_StateModel):
    item_id: str
    kind: str
    content: str | None = None
    reference: str | None = None
    source_event_id: str | None = None
    source_event_sequence: int | None = None
    context_id: str | None = None
    source: str = "unknown"
    source_channel: str = "unknown"
    source_session_id: str | None = None
    activation: float = Field(ge=0.0, le=1.0)
    salience: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    last_accessed_at: datetime
    retention_reason: str


class WorkingMemoryStateSnapshot(_StateModel):
    items: list[WorkingMemoryItemSnapshot] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)


class MotivationStateSnapshot(_StateModel):
    active_goals: list[dict[str, Any]] = Field(default_factory=list)
    commitments: list[dict[str, Any]] = Field(default_factory=list)
    extensions: dict[str, Any] = Field(default_factory=dict)


class IdentityStateSnapshot(_StateModel):
    values: dict[str, Any] = Field(default_factory=dict)
    self_model: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)


class ContextFrameSnapshot(_StateModel):
    context_id: str
    context_type: str
    source_channel: str
    source_session_id: str | None = None
    participant_ids: list[str] = Field(default_factory=list)
    active_topic: str | None = None
    active_task: str | None = None
    started_at: datetime
    last_active_at: datetime
    parent_context_id: str | None = None
    related_context_ids: list[str] = Field(default_factory=list)
    status: str


class InterlocutorSnapshot(_StateModel):
    identity_key: str
    relationship_metadata: dict[str, Any] = Field(default_factory=dict)
    shared_history_references: list[str] = Field(default_factory=list)
    inferred_preferences: dict[str, dict[str, Any]] = Field(default_factory=dict)
    uncertainties: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ContextStateSnapshot(_StateModel):
    frames: list[ContextFrameSnapshot] = Field(default_factory=list)
    interlocutors: list[InterlocutorSnapshot] = Field(default_factory=list)


class AgentStateSnapshot(_StateModel):
    schema_version: Literal[3] = 3
    saved_at: datetime
    last_processed_event_sequence: int = Field(ge=0)
    emotion_state: EmotionStateSnapshot
    working_memory: WorkingMemoryStateSnapshot = Field(
        default_factory=WorkingMemoryStateSnapshot
    )
    motivation: MotivationStateSnapshot = Field(default_factory=MotivationStateSnapshot)
    identity: IdentityStateSnapshot = Field(default_factory=IdentityStateSnapshot)
    context_state: ContextStateSnapshot = Field(default_factory=ContextStateSnapshot)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_private_runtime_fields(self) -> "AgentStateSnapshot":
        forbidden = {"hidden_thought", "prompt", "turns", "attachments", "event_payload"}
        if _contains_forbidden_key(self.model_dump(), forbidden):
            raise ValueError("snapshot contains a forbidden private runtime field")
        if self.saved_at.tzinfo is None:
            raise ValueError("saved_at must include a timezone")
        return self


@dataclass
class PersistentAgentState:
    working_memory_metadata: dict[str, Any] = field(default_factory=dict)
    working_memory_extensions: dict[str, Any] = field(default_factory=dict)
    active_goals: list[dict[str, Any]] = field(default_factory=list)
    commitments: list[dict[str, Any]] = field(default_factory=list)
    motivation_extensions: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    self_model: dict[str, Any] = field(default_factory=dict)
    identity_extensions: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)


class AgentStateStore:
    def __init__(self, path: Path, event_recorder: Any | None = None) -> None:
        self.path = path
        self.event_recorder = event_recorder
        self.last_snapshot: AgentStateSnapshot | None = None

    def load(self, baseline_surprisal: float) -> AgentStateSnapshot:
        default = default_agent_state_snapshot(baseline_surprisal)
        if not self.path.exists():
            self.last_snapshot = default
            return default
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("snapshot root must be an object")
            version = raw.get("schema_version")
            if version == 0:
                raw = _migrate_v0(raw)
                self._record("migrated", {"from_version": 0, "to_version": 3})
            elif version == 1:
                raw = _migrate_v1(raw)
                self._record("migrated", {"from_version": 1, "to_version": 3})
            elif version == 2:
                raw = _migrate_v2(raw)
                self._record("migrated", {"from_version": 2, "to_version": 3})
            elif version != CURRENT_AGENT_STATE_SCHEMA_VERSION:
                raise UnsupportedStateVersion(version)
            snapshot = AgentStateSnapshot.model_validate(raw)
        except UnsupportedStateVersion as exc:
            self._record("load_failed", {"reason": "unsupported_schema_version", "schema_version": exc.version})
            snapshot = default
        except json.JSONDecodeError:
            self._record("load_failed", {"reason": "json_decode_error"})
            snapshot = default
        except (OSError, ValueError, ValidationError):
            self._record("load_failed", {"reason": "validation_or_io_error"})
            snapshot = default
        self.last_snapshot = snapshot
        return snapshot

    def save(self, snapshot: AgentStateSnapshot) -> AgentStateSnapshot:
        validated = AgentStateSnapshot.model_validate(snapshot.model_dump())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as snapshot_file:
                json.dump(validated.model_dump(mode="json"), snapshot_file, ensure_ascii=True, sort_keys=True)
                snapshot_file.flush()
                os.fsync(snapshot_file.fileno())
            os.replace(temporary_path, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)
        self.last_snapshot = validated
        return validated

    def capture(self, main_loop: Any, sequence: int) -> AgentStateSnapshot:
        state = main_loop.persistent_state
        emotion = main_loop.emotion_engine.state
        return AgentStateSnapshot(
            saved_at=datetime.now(UTC),
            last_processed_event_sequence=sequence,
            emotion_state=EmotionStateSnapshot(
                valence=emotion.valence,
                arousal=emotion.arousal,
                optimal_loss=emotion.optimal_loss,
            ),
            working_memory=WorkingMemoryStateSnapshot(
                items=[_working_memory_item_snapshot(item) for item in main_loop.working_memory.items],
                metadata=state.working_memory_metadata,
                extensions=state.working_memory_extensions,
            ),
            motivation=MotivationStateSnapshot(
                active_goals=state.active_goals,
                commitments=state.commitments,
                extensions=state.motivation_extensions,
            ),
            identity=IdentityStateSnapshot(
                values=state.values,
                self_model=state.self_model,
                extensions=state.identity_extensions,
            ),
            context_state=ContextStateSnapshot(
                frames=[_context_frame_snapshot(frame) for frame in main_loop.context_registry.frames],
                interlocutors=[
                    _interlocutor_snapshot(model)
                    for model in main_loop.context_registry.interlocutors
                ],
            ),
            extensions=state.extensions,
        )

    def restore_into(self, main_loop: Any, snapshot: AgentStateSnapshot) -> None:
        main_loop.emotion_engine.state = EmotionState(**snapshot.emotion_state.model_dump())
        main_loop.persistent_state = persistent_state_from_snapshot(snapshot)
        main_loop.restore_appraisal_state()
        main_loop.restore_value_state()
        main_loop.restore_motivation_state()
        main_loop.restore_decision_state()
        main_loop.restore_self_model_state()
        main_loop.restore_experience_state()
        main_loop.restore_belief_state()
        main_loop.working_memory.restore(
            [_working_memory_item_from_snapshot(item) for item in snapshot.working_memory.items]
        )
        main_loop._sync_belief_working_memory(None)
        main_loop.context_registry.restore(
            tuple(_context_frame_from_snapshot(item) for item in snapshot.context_state.frames),
            tuple(
                _interlocutor_from_snapshot(item)
                for item in snapshot.context_state.interlocutors
            ),
        )

    def save_failed_sequence(self, sequence: int) -> AgentStateSnapshot | None:
        if self.last_snapshot is None:
            return None
        return self.save(
            self.last_snapshot.model_copy(
                update={
                    "saved_at": datetime.now(UTC),
                    "last_processed_event_sequence": sequence,
                }
            )
        )

    def _record(self, event_type: str, metadata: dict[str, Any]) -> None:
        if self.event_recorder is not None:
            self.event_recorder.record(
                category="state",
                event_type=event_type,
                message="Agent state snapshot load event",
                metadata=metadata,
            )


class UnsupportedStateVersion(ValueError):
    def __init__(self, version: object) -> None:
        self.version = version
        super().__init__(f"Unsupported agent state schema version: {version}")


def default_agent_state_snapshot(baseline_surprisal: float) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        saved_at=datetime.now(UTC),
        last_processed_event_sequence=0,
        emotion_state=EmotionStateSnapshot(
            valence=0.0, arousal=0.0, optimal_loss=baseline_surprisal
        ),
    )


def persistent_state_from_snapshot(snapshot: AgentStateSnapshot) -> PersistentAgentState:
    return PersistentAgentState(
        working_memory_metadata=dict(snapshot.working_memory.metadata),
        working_memory_extensions=dict(snapshot.working_memory.extensions),
        active_goals=list(snapshot.motivation.active_goals),
        commitments=list(snapshot.motivation.commitments),
        motivation_extensions=dict(snapshot.motivation.extensions),
        values=dict(snapshot.identity.values),
        self_model=dict(snapshot.identity.self_model),
        identity_extensions=dict(snapshot.identity.extensions),
        extensions=dict(snapshot.extensions),
    )


def _migrate_v0(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "saved_at": datetime.now(UTC).isoformat(),
        "last_processed_event_sequence": raw.get("last_event_sequence", 0),
        "emotion_state": raw.get("emotion", {}),
        "working_memory": {},
        "motivation": {},
        "identity": {},
        "context_state": {},
        "extensions": {},
    }


def _migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(raw)
    migrated["schema_version"] = 3
    working_memory = dict(migrated.get("working_memory") or {})
    working_memory["items"] = []
    migrated["working_memory"] = working_memory
    migrated["context_state"] = {}
    return migrated


def _migrate_v2(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(raw)
    migrated["schema_version"] = 3
    migrated["context_state"] = {}
    return migrated


def _working_memory_item_snapshot(item: WorkingMemoryItem) -> WorkingMemoryItemSnapshot:
    return WorkingMemoryItemSnapshot(
        item_id=item.item_id,
        kind=item.kind.value,
        content=None if item.reference is not None else item.content,
        reference=item.reference,
        source_event_id=item.source_event_id,
        source_event_sequence=item.source_event_sequence,
        context_id=item.context_id,
        source=item.source,
        source_channel=item.source_channel,
        source_session_id=item.source_session_id,
        activation=item.activation,
        salience=item.salience,
        created_at=item.created_at,
        last_accessed_at=item.last_accessed_at,
        retention_reason=item.retention_reason.value,
    )


def _working_memory_item_from_snapshot(
    item: WorkingMemoryItemSnapshot,
) -> WorkingMemoryItem:
    return WorkingMemoryItem(
        item_id=item.item_id,
        kind=WorkingMemoryKind(item.kind),
        content=item.content,
        reference=item.reference,
        source_event_id=item.source_event_id,
        source_event_sequence=item.source_event_sequence,
        context_id=item.context_id,
        source=item.source,
        source_channel=item.source_channel,
        source_session_id=item.source_session_id,
        activation=item.activation,
        salience=item.salience,
        created_at=item.created_at,
        last_accessed_at=item.last_accessed_at,
        retention_reason=RetentionReason(item.retention_reason),
    )


def _context_frame_snapshot(frame: ContextFrame) -> ContextFrameSnapshot:
    return ContextFrameSnapshot(
        context_id=frame.context_id,
        context_type=frame.context_type,
        source_channel=frame.source_channel,
        source_session_id=frame.source_session_id,
        participant_ids=list(frame.participant_ids),
        active_topic=frame.active_topic,
        active_task=frame.active_task,
        started_at=frame.started_at,
        last_active_at=frame.last_active_at,
        parent_context_id=frame.parent_context_id,
        related_context_ids=list(frame.related_context_ids),
        status=frame.status.value,
    )


def _context_frame_from_snapshot(item: ContextFrameSnapshot) -> ContextFrame:
    return ContextFrame(
        context_id=item.context_id,
        context_type=item.context_type,
        source_channel=item.source_channel,
        source_session_id=item.source_session_id,
        participant_ids=tuple(item.participant_ids),
        active_topic=item.active_topic,
        active_task=item.active_task,
        started_at=item.started_at,
        last_active_at=item.last_active_at,
        parent_context_id=item.parent_context_id,
        related_context_ids=tuple(item.related_context_ids),
        status=ContextStatus(item.status),
    )


def _interlocutor_snapshot(model: InterlocutorModel) -> InterlocutorSnapshot:
    return InterlocutorSnapshot(
        identity_key=model.identity_key,
        relationship_metadata=model.relationship_metadata,
        shared_history_references=list(model.shared_history_references),
        inferred_preferences={
            key: _attribute_json(value)
            for key, value in model.inferred_preferences.items()
        },
        uncertainties={
            key: _attribute_json(value) for key, value in model.uncertainties.items()
        },
    )


def _interlocutor_from_snapshot(item: InterlocutorSnapshot) -> InterlocutorModel:
    return InterlocutorModel(
        identity_key=item.identity_key,
        relationship_metadata=item.relationship_metadata,
        shared_history_references=tuple(item.shared_history_references),
        inferred_preferences={
            key: _attribute_from_json(value)
            for key, value in item.inferred_preferences.items()
        },
        uncertainties={
            key: _attribute_from_json(value)
            for key, value in item.uncertainties.items()
        },
    )


def _attribute_json(attribute: InferredAttribute) -> dict[str, Any]:
    return {
        "value": attribute.value,
        "confidence": attribute.confidence,
        "evidence_references": list(attribute.evidence_references),
    }


def _attribute_from_json(value: dict[str, Any]) -> InferredAttribute:
    return InferredAttribute(
        value=value.get("value"),
        confidence=float(value.get("confidence", 0.0)),
        evidence_references=tuple(
            str(item) for item in value.get("evidence_references", [])
        ),
    )


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden
            or _contains_forbidden_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False

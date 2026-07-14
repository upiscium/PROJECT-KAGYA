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


CURRENT_AGENT_STATE_SCHEMA_VERSION = 1


class _StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmotionStateSnapshot(_StateModel):
    valence: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    arousal: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    optimal_loss: float = Field(ge=0.0, allow_inf_nan=False)


class WorkingMemoryStateSnapshot(_StateModel):
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


class AgentStateSnapshot(_StateModel):
    schema_version: Literal[1] = CURRENT_AGENT_STATE_SCHEMA_VERSION
    saved_at: datetime
    last_processed_event_sequence: int = Field(ge=0)
    emotion_state: EmotionStateSnapshot
    working_memory: WorkingMemoryStateSnapshot = Field(
        default_factory=WorkingMemoryStateSnapshot
    )
    motivation: MotivationStateSnapshot = Field(default_factory=MotivationStateSnapshot)
    identity: IdentityStateSnapshot = Field(default_factory=IdentityStateSnapshot)
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
                self._record("migrated", {"from_version": 0, "to_version": 1})
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
            extensions=state.extensions,
        )

    def restore_into(self, main_loop: Any, snapshot: AgentStateSnapshot) -> None:
        main_loop.emotion_engine.state = EmotionState(**snapshot.emotion_state.model_dump())
        main_loop.persistent_state = persistent_state_from_snapshot(snapshot)

    def save_failed_sequence(self, sequence: int) -> None:
        if self.last_snapshot is None:
            return
        self.save(
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
        "schema_version": 1,
        "saved_at": datetime.now(UTC).isoformat(),
        "last_processed_event_sequence": raw.get("last_event_sequence", 0),
        "emotion_state": raw.get("emotion", {}),
        "working_memory": {},
        "motivation": {},
        "identity": {},
        "extensions": {},
    }


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

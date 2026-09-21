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


CURRENT_AGENT_STATE_SCHEMA_VERSION: Literal[4] = 4


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


class AgentStateSnapshot(_AgentStateSnapshotBase):
    """Current AgentState v4 canonical snapshot."""

    schema_version: Literal[4] = CURRENT_AGENT_STATE_SCHEMA_VERSION
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot
    appraisal_state: AppraisalStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


CompatibleAgentStateSnapshot = Annotated[
    AgentStateSnapshotV1
    | AgentStateSnapshotV2
    | AgentStateSnapshotV3
    | AgentStateSnapshot,
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


def default_agent_state_snapshot(
    baseline_surprisal: float,
    *,
    saved_at: datetime | None = None,
) -> AgentStateSnapshot:
    """Return the bootstrap state used only when the canonical file is absent."""

    return AgentStateSnapshot(
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
    )


class AgentStateStore:
    """Load, capture, restore, and atomically publish the R04 snapshot."""

    def __init__(
        self,
        path: str | Path,
        baseline_surprisal: float,
        *,
        clock: Callable[[], datetime] | None = None,
        save_stage_hook: Callable[[AgentStateSaveStage], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._baseline_surprisal = baseline_surprisal
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._save_stage_hook = save_stage_hook

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
                    self._baseline_surprisal, saved_at=self._now()
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
        if version == CURRENT_AGENT_STATE_SCHEMA_VERSION:
            schema_failure = None
            try:
                return AgentStateSnapshot.model_validate(raw)
            except ValidationError:
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
            return self._canonical_bytes(validated)
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
                    (AgentStateSnapshotV1, AgentStateSnapshotV2, AgentStateSnapshotV3),
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

    def capture(self, main_loop: KagyaMainLoop, sequence: int) -> AgentStateSnapshot:
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
            return AgentStateSnapshot(
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
        emotion_engine: EmotionEngineAllostasis | None = None
        working_memory_authority: WorkingMemory | None = None
        try:
            validated = validate_compatible_agent_state_snapshot(
                snapshot.model_dump(mode="python")
            )
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
                    (AgentStateSnapshotV2, AgentStateSnapshotV3, AgentStateSnapshot),
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
                if isinstance(validated, (AgentStateSnapshotV3, AgentStateSnapshot))
                else ContextRegistryState(0, None, (), ())
            )
            validate_context_registry_state(context_state)
            calibration = getattr(main_loop, "loss_calibration", None)
            if not isinstance(calibration, LossCalibration):
                raise AgentStateLoadError("AgentState restore requires LossCalibration")
            appraisal = (
                validated.appraisal_state
                if isinstance(validated, AgentStateSnapshot)
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
                isinstance(validated, (AgentStateSnapshotV3, AgentStateSnapshot))
                and context_registry is None
            ):
                raise AgentStateLoadError("AgentState restore requires ContextRegistry")

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
            emotion_engine.state = EmotionState(
                valence=emotion.valence,
                arousal=emotion.arousal,
                optimal_loss=emotion.optimal_loss,
            )
            emotion_engine.temporal_state = restored_temporal
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
            restore_failure = AgentStateLoadError("AgentState restore failed")
        if restore_failure is not None:
            raise restore_failure

    def _migrate_v0(self, raw: dict[str, Any]) -> AgentStateSnapshot:
        migration_failure: AgentStateLoadError | None = None
        try:
            legacy = _LegacyAgentStateV0.model_validate(raw)
            return AgentStateSnapshot(
                saved_at=self._now(),
                last_processed_event_sequence=legacy.last_event_sequence,
                emotion_state=EmotionStateSnapshot(
                    valence=legacy.emotion.valence,
                    arousal=legacy.emotion.arousal,
                    optimal_loss=legacy.emotion.optimal_loss,
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

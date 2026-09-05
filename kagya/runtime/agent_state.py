"""Versioned, minimal, durable AgentState snapshot authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kagya.body import EmotionState
from kagya.privacy import normalize_private_key

if TYPE_CHECKING:
    from kagya.runtime.main_loop import KagyaMainLoop


CURRENT_AGENT_STATE_SCHEMA_VERSION: Literal[1] = 1


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


class AgentStateSnapshot(_StateModel):
    schema_version: Literal[1] = CURRENT_AGENT_STATE_SCHEMA_VERSION
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
        "prompt",
        "rawprompt",
        "systemprompt",
        "userprompt",
        "assistantprompt",
        "retrievedmemory",
        "privatestate",
        "turns",
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

    def load(self) -> AgentStateSnapshot:
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
        if version == CURRENT_AGENT_STATE_SCHEMA_VERSION:
            schema_failure: AgentStateLoadError | None = None
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

    def save(self, snapshot: AgentStateSnapshot) -> None:
        stage = AgentStateSaveStage.TEMP_WRITE
        published = False
        temporary_path: Path | None = None
        descriptor: int | None = None
        save_failure: AgentStateSaveError | None = None
        try:
            raw: object = (
                snapshot.model_dump(mode="python")
                if isinstance(snapshot, AgentStateSnapshot)
                else snapshot
            )
            _reject_private_keys(raw)
            validated = AgentStateSnapshot.model_validate(raw)
            payload = self._canonical_bytes(validated)
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

    def capture(self, main_loop: KagyaMainLoop, sequence: int) -> AgentStateSnapshot:
        capture_failure: AgentStateSaveError | None = None
        try:
            emotion = main_loop.emotion_engine.state
            return AgentStateSnapshot(
                saved_at=self._now(),
                last_processed_event_sequence=sequence,
                emotion_state=EmotionStateSnapshot(
                    valence=emotion.valence,
                    arousal=emotion.arousal,
                    optimal_loss=emotion.optimal_loss,
                ),
            )
        except Exception:
            capture_failure = AgentStateSaveError(
                AgentStateSaveStage.CAPTURE, published=False
            )
        raise capture_failure

    def restore_into(
        self, main_loop: KagyaMainLoop, snapshot: AgentStateSnapshot
    ) -> None:
        restore_failure: AgentStateLoadError | None = None
        try:
            validated = AgentStateSnapshot.model_validate(
                snapshot.model_dump(mode="python")
            )
            emotion = validated.emotion_state
            main_loop.emotion_engine.state = EmotionState(
                valence=emotion.valence,
                arousal=emotion.arousal,
                optimal_loss=emotion.optimal_loss,
            )
        except Exception:
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
    def _canonical_bytes(snapshot: AgentStateSnapshot) -> bytes:
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

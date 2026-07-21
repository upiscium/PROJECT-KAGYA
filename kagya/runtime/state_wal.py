"""Private write-ahead log for deterministic authoritative state reconstruction."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.runtime.agent_state import AgentStateSnapshot
from kagya.runtime.event_journal import hash_snapshot


class StateWalIntegrityError(RuntimeError):
    """Raised when private state transition history cannot be trusted."""


class StateWalEvent(Protocol):
    event_id: str
    event_type: Any
    source: str
    processing_sequence: int | None


class StatePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["replace"] = "replace"
    path: Literal[""] = ""
    value: AgentStateSnapshot


class StateWalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    command_version: Literal[1] = 1
    command_type: Literal["state_patch"] = "state_patch"
    record_id: str
    timestamp: datetime
    event_id: str
    event_type: str
    source: str
    processing_sequence: int = Field(ge=0)
    patch: StatePatch
    state_hash_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_hash_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_record_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_transition(self) -> "StateWalRecord":
        if self.timestamp.tzinfo is None:
            raise ValueError("WAL timestamp must include a timezone")
        if self.patch.value.last_processed_event_sequence != self.processing_sequence:
            raise ValueError("WAL patch sequence does not match its record")
        if hash_snapshot(self.patch.value) != self.state_hash_after:
            raise ValueError("WAL patch does not match its post-state hash")
        _reject_private_fields(self.patch.value.model_dump(mode="json"))
        return self


class StateDiffEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    before: Any = None
    after: Any = None


class StateReconstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: AgentStateSnapshot
    external_side_effects_replayed: Literal[False] = False


class StateDryRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_sequence: int = Field(ge=0)
    target_sequence: int = Field(ge=0)
    current_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    changes: list[StateDiffEntry]
    external_side_effects_replayed: Literal[False] = False


class StateWAL:
    """Store validated state replacement patches in a private hash-chained file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        records = self.verify()
        self._last_hash = records[-1].record_hash if records else None
        self._last_sequence = records[-1].processing_sequence if records else None
        self._last_state_hash = records[-1].state_hash_after if records else None

    def bootstrap(self, snapshot: AgentStateSnapshot) -> StateWalRecord | None:
        with self._lock:
            if self._last_hash is not None:
                return None
            snapshot_hash = hash_snapshot(snapshot)
            return self._append_locked(
                event_id="state-wal-bootstrap",
                event_type="state_snapshot",
                source="state_wal.bootstrap",
                processing_sequence=snapshot.last_processed_event_sequence,
                before_hash=snapshot_hash,
                snapshot=snapshot,
            )

    def append_transition(
        self,
        event: StateWalEvent,
        before: AgentStateSnapshot,
        after: AgentStateSnapshot,
    ) -> StateWalRecord:
        sequence = event.processing_sequence
        if sequence is None:
            raise StateWalIntegrityError("WAL event has no processing sequence")
        before_hash = hash_snapshot(before)
        with self._lock:
            if self._last_sequence is None or self._last_state_hash is None:
                raise StateWalIntegrityError("WAL has no authoritative baseline")
            if sequence != self._last_sequence + 1:
                raise StateWalIntegrityError(
                    "WAL processing sequence is not contiguous"
                )
            if before_hash != self._last_state_hash:
                raise StateWalIntegrityError(
                    "WAL pre-state hash does not match current state"
                )
            event_type = getattr(event.event_type, "value", str(event.event_type))
            return self._append_locked(
                event_id=event.event_id,
                event_type=str(event_type),
                source=event.source,
                processing_sequence=sequence,
                before_hash=before_hash,
                snapshot=after,
            )

    def reconstruct(self, sequence: int | None = None) -> StateReconstruction:
        records = self.verify()
        if not records:
            raise StateWalIntegrityError("WAL has no authoritative baseline")
        target = (
            records[-1]
            if sequence is None
            else next(
                (
                    record
                    for record in records
                    if record.processing_sequence == sequence
                ),
                None,
            )
        )
        if target is None:
            raise StateWalIntegrityError(f"WAL sequence {sequence} is not retained")
        snapshot = AgentStateSnapshot.model_validate(target.patch.value.model_dump())
        return StateReconstruction(
            sequence=target.processing_sequence,
            snapshot_hash=target.state_hash_after,
            snapshot=snapshot,
        )

    def dry_run(self, current: AgentStateSnapshot, target_sequence: int) -> StateDryRun:
        target = self.reconstruct(target_sequence)
        return StateDryRun(
            current_sequence=current.last_processed_event_sequence,
            target_sequence=target.sequence,
            current_hash=hash_snapshot(current),
            target_hash=target.snapshot_hash,
            changes=_state_diff(
                current.model_dump(mode="json"), target.snapshot.model_dump(mode="json")
            ),
        )

    def verify(self) -> list[StateWalRecord]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise StateWalIntegrityError("private state WAL cannot be read") from exc
        records: list[StateWalRecord] = []
        previous_hash: str | None = None
        previous_sequence: int | None = None
        previous_state_hash: str | None = None
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = StateWalRecord.model_validate_json(line)
            except ValueError as exc:
                raise StateWalIntegrityError(
                    f"private state WAL record is invalid at line {line_number}"
                ) from exc
            if _record_hash(record) != record.record_hash:
                raise StateWalIntegrityError("private state WAL record hash mismatch")
            if record.previous_record_hash != previous_hash:
                raise StateWalIntegrityError("private state WAL hash chain is broken")
            if previous_sequence is None:
                if record.state_hash_before != record.state_hash_after:
                    raise StateWalIntegrityError(
                        "private state WAL baseline hash is invalid"
                    )
            else:
                if record.processing_sequence != previous_sequence + 1:
                    raise StateWalIntegrityError("private state WAL sequence gap")
                if record.state_hash_before != previous_state_hash:
                    raise StateWalIntegrityError(
                        "private state WAL state hash chain is broken"
                    )
            previous_hash = record.record_hash
            previous_sequence = record.processing_sequence
            previous_state_hash = record.state_hash_after
            records.append(record)
        return records

    def _append_locked(
        self,
        *,
        event_id: str,
        event_type: str,
        source: str,
        processing_sequence: int,
        before_hash: str,
        snapshot: AgentStateSnapshot,
    ) -> StateWalRecord:
        _reject_private_fields(snapshot.model_dump(mode="json"))
        provisional = StateWalRecord(
            record_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            event_id=event_id,
            event_type=event_type,
            source=source,
            processing_sequence=processing_sequence,
            patch=StatePatch(value=snapshot),
            state_hash_before=before_hash,
            state_hash_after=hash_snapshot(snapshot),
            previous_record_hash=self._last_hash,
            record_hash="0" * 64,
        )
        record = provisional.model_copy(
            update={"record_hash": _record_hash(provisional)}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
        except Exception:
            os.close(descriptor)
            raise
        with os.fdopen(descriptor, "a", encoding="utf-8") as output:
            output.write(record.model_dump_json() + "\n")
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(self.path.parent)
        self._last_hash = record.record_hash
        self._last_sequence = processing_sequence
        self._last_state_hash = record.state_hash_after
        return record


def _record_hash(record: StateWalRecord) -> str:
    canonical = json.dumps(
        record.model_dump(mode="json", exclude={"record_hash"}),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _reject_private_fields(value: Any) -> None:
    forbidden = {
        "attachmentbody",
        "attachments",
        "credential",
        "credentials",
        "eventpayload",
        "hiddenthought",
        "password",
        "prompt",
        "rawprompt",
        "secret",
        "token",
        "turns",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(
                character for character in str(key).lower() if character.isalnum()
            )
            if normalized in forbidden:
                raise StateWalIntegrityError("private field is forbidden in state WAL")
            _reject_private_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_fields(item)


def _state_diff(before: Any, after: Any, path: str = "") -> list[StateDiffEntry]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[StateDiffEntry] = []
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}/{key}"
            if key not in before:
                changes.append(StateDiffEntry(path=child_path, after=after[key]))
            elif key not in after:
                changes.append(StateDiffEntry(path=child_path, before=before[key]))
            else:
                changes.extend(_state_diff(before[key], after[key], child_path))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        if before == after:
            return []
        return [StateDiffEntry(path=path or "/", before=before, after=after)]
    if before == after:
        return []
    return [StateDiffEntry(path=path or "/", before=before, after=after)]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

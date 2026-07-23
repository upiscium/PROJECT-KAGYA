"""Durable operator-safe lifecycle journal for authoritative agent events."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.runtime.agent_state import AgentStateSnapshot


class JournalIntegrityError(RuntimeError):
    """Raised when durable journal or snapshot continuity cannot be trusted."""


class JournalLifecycle(StrEnum):
    ACCEPTED = "accepted"
    STARTED = "started"
    PREPARED = "prepared"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERY_CLASSIFIED = "recovery_classified"
    CHECKPOINT = "checkpoint"
    AUDIT = "audit"


class JournalEvent(Protocol):
    event_id: str
    event_type: Any
    source: str
    processing_sequence: int | None
    causation_id: str | None
    correlation_id: str | None


class JournalTelemetry(Protocol):
    def storage_observation(
        self, component: str, operation: str, status: str, duration: float
    ) -> None: ...


class JournalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2] = 2
    record_id: str
    timestamp: datetime
    lifecycle: JournalLifecycle
    event_id: str
    event_type: str
    source: str
    processing_sequence: int | None = Field(default=None, ge=0)
    snapshot_sequence: int | None = Field(default=None, ge=0)
    causation_id: str | None = None
    correlation_id: str | None = None
    state_hash_before: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    state_hash_after: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_category: str | None = None
    actor_id: str | None = None
    actor_role: str | None = None
    target: str | None = None
    reauthenticated: bool | None = None
    previous_record_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lifecycle_fields(self) -> "JournalRecord":
        if self.timestamp.tzinfo is None:
            raise ValueError("journal timestamp must include a timezone")
        if (
            self.lifecycle == JournalLifecycle.ACCEPTED
            and self.processing_sequence is not None
        ):
            raise ValueError("accepted journal record must not have a sequence")
        if (
            self.lifecycle
            in {
                JournalLifecycle.STARTED,
                JournalLifecycle.PREPARED,
                JournalLifecycle.COMPLETED,
                JournalLifecycle.FAILED,
            }
            and self.processing_sequence is None
        ):
            raise ValueError("journal lifecycle record requires a sequence")
        if self.lifecycle == JournalLifecycle.PREPARED and (
            self.state_hash_before is None or self.state_hash_after is None
        ):
            raise ValueError("prepared journal record requires state hashes")
        if self.lifecycle in {
            JournalLifecycle.COMPLETED,
            JournalLifecycle.FAILED,
        } and (self.snapshot_hash is None or self.snapshot_sequence is None):
            raise ValueError("terminal journal record requires snapshot continuity")
        if (
            self.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
            and self.failure_category is None
        ):
            raise ValueError("recovery journal record requires a classification")
        if self.lifecycle == JournalLifecycle.AUDIT and (
            self.actor_id is None or self.actor_role is None or self.target is None
        ):
            raise ValueError("audit journal record requires actor and target")
        return self


class EventJournal:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 10 * 1024 * 1024,
        retained_files: int = 3,
        telemetry: JournalTelemetry | None = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("journal max_bytes must be positive")
        if retained_files < 1:
            raise ValueError("journal retained_files must be at least one")
        self.path = path
        self.max_bytes = max_bytes
        self.retained_files = retained_files
        self._telemetry = telemetry
        self._lock = Lock()
        self._last_hash: str | None = None
        self._last_sequence = 0
        self._last_snapshot_hash: str | None = None
        self._last_snapshot_sequence = 0
        records = self.verify()
        if records:
            self._last_hash = records[-1].record_hash
            self._last_sequence = max(
                (record.processing_sequence or 0 for record in records),
                default=0,
            )
            snapshot_records = [
                record
                for record in records
                if record.snapshot_hash is not None
                and record.snapshot_sequence is not None
            ]
            if snapshot_records:
                self._last_snapshot_hash = snapshot_records[-1].snapshot_hash
                self._last_snapshot_sequence = (
                    snapshot_records[-1].snapshot_sequence or 0
                )

    def accepted(self, event: JournalEvent) -> JournalRecord:
        return self._append_event(event, JournalLifecycle.ACCEPTED)

    def started(self, event: JournalEvent) -> JournalRecord:
        return self._append_event(event, JournalLifecycle.STARTED)

    def prepared(
        self,
        event: JournalEvent,
        *,
        state_hash_before: str,
        state_hash_after: str,
    ) -> JournalRecord:
        return self._append_event(
            event,
            JournalLifecycle.PREPARED,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
        )

    def completed(self, event: JournalEvent, snapshot_hash: str) -> JournalRecord:
        return self._append_event(
            event,
            JournalLifecycle.COMPLETED,
            snapshot_hash=snapshot_hash,
            snapshot_sequence=event.processing_sequence,
        )

    def failed(
        self,
        event: JournalEvent,
        failure_category: str,
        snapshot_hash: str | None,
    ) -> JournalRecord:
        return self._append_event(
            event,
            JournalLifecycle.FAILED,
            snapshot_hash=snapshot_hash,
            snapshot_sequence=event.processing_sequence,
            failure_category=_safe_label(failure_category),
        )

    def recent(self, limit: int = 50) -> list[JournalRecord]:
        if limit <= 0:
            return []
        with self._lock:
            return self.verify()[-limit:]

    def audit_admin_action(
        self,
        *,
        event_id: str,
        actor_id: str,
        actor_role: str,
        target: str,
        reauthenticated: bool,
    ) -> JournalRecord:
        return self._append(
            lifecycle=JournalLifecycle.AUDIT,
            event_id=_safe_label(event_id),
            event_type="admin_action",
            source="api.admin",
            actor_id=_safe_label(actor_id),
            actor_role=_safe_label(actor_role),
            target=_safe_target(target),
            reauthenticated=reauthenticated,
        )

    def reconcile(self, snapshot: AgentStateSnapshot) -> list[JournalRecord]:
        snapshot_hash = hash_snapshot(snapshot)
        records = self.verify()
        if not records:
            if snapshot.last_processed_event_sequence:
                return [self._checkpoint(snapshot)]
            return []

        latest_by_event: dict[str, JournalRecord] = {}
        for record in records:
            if record.lifecycle != JournalLifecycle.CHECKPOINT:
                latest_by_event[record.event_id] = record

        terminal = [
            record
            for record in records
            if (
                record.lifecycle
                in {
                    JournalLifecycle.COMPLETED,
                    JournalLifecycle.FAILED,
                }
                or (
                    record.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                    and record.failure_category == "committed_before_crash"
                )
                or (
                    record.lifecycle == JournalLifecycle.CHECKPOINT
                    and record.snapshot_hash is not None
                )
            )
            and record.snapshot_sequence is not None
        ]
        last_terminal = terminal[-1] if terminal else None
        if last_terminal is not None:
            terminal_sequence = last_terminal.snapshot_sequence
            if terminal_sequence is None:
                raise JournalIntegrityError(
                    "terminal journal record has no snapshot sequence"
                )
            if terminal_sequence > snapshot.last_processed_event_sequence:
                raise JournalIntegrityError("journal sequence is ahead of the snapshot")
            if (
                terminal_sequence == snapshot.last_processed_event_sequence
                and last_terminal.snapshot_hash is not None
                and last_terminal.snapshot_hash != snapshot_hash
            ):
                raise JournalIntegrityError("journal and snapshot hashes disagree")
            if terminal_sequence < snapshot.last_processed_event_sequence and not any(
                record.lifecycle == JournalLifecycle.PREPARED
                and record.processing_sequence == snapshot.last_processed_event_sequence
                and record.state_hash_after == snapshot_hash
                for record in latest_by_event.values()
            ):
                raise JournalIntegrityError("snapshot sequence is ahead of the journal")
        elif snapshot.last_processed_event_sequence and not any(
            record.lifecycle == JournalLifecycle.PREPARED
            and record.processing_sequence == snapshot.last_processed_event_sequence
            and record.state_hash_after == snapshot_hash
            for record in latest_by_event.values()
        ):
            raise JournalIntegrityError("snapshot has no journal commit evidence")
        recovered: list[JournalRecord] = []
        for record in latest_by_event.values():
            if record.lifecycle == JournalLifecycle.ACCEPTED:
                recovered.append(
                    self._recovery_record(
                        record,
                        "accepted_not_started",
                        snapshot_hash,
                        snapshot.last_processed_event_sequence,
                    )
                )
            elif record.lifecycle in {
                JournalLifecycle.STARTED,
                JournalLifecycle.PREPARED,
            }:
                recovered.append(
                    self._classify_interrupted(record, snapshot, snapshot_hash)
                )
        return recovered

    def verify(self) -> list[JournalRecord]:
        records: list[JournalRecord] = []
        previous_hash: str | None = None
        sequence_anchor = 0
        has_sequence_anchor = False
        for path in self._journal_paths():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                raise JournalIntegrityError(f"journal cannot be read: {path}") from exc
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = JournalRecord.model_validate_json(line)
                except ValueError as exc:
                    raise JournalIntegrityError(
                        f"journal record is invalid: {path}:{line_number}"
                    ) from exc
                if _record_hash(record) != record.record_hash:
                    raise JournalIntegrityError("journal record hash mismatch")
                if (
                    previous_hash is not None
                    and record.previous_record_hash != previous_hash
                ):
                    raise JournalIntegrityError("journal hash chain is broken")
                if (
                    previous_hash is None
                    and records
                    and record.lifecycle != JournalLifecycle.CHECKPOINT
                ):
                    raise JournalIntegrityError("rotated journal has no checkpoint")
                if (
                    record.lifecycle == JournalLifecycle.CHECKPOINT
                    and record.processing_sequence is not None
                ):
                    sequence_anchor = record.processing_sequence
                    has_sequence_anchor = True
                elif record.lifecycle == JournalLifecycle.STARTED:
                    if record.processing_sequence is None:
                        raise JournalIntegrityError("started event has no sequence")
                    expected = sequence_anchor + 1
                    if record.processing_sequence != expected:
                        raise JournalIntegrityError("journal processing sequence gap")
                    sequence_anchor = record.processing_sequence
                    has_sequence_anchor = True
                elif (
                    record.processing_sequence is not None
                    and has_sequence_anchor
                    and record.processing_sequence > sequence_anchor
                ):
                    raise JournalIntegrityError(
                        "journal lifecycle record precedes its started event"
                    )
                previous_hash = record.record_hash
                records.append(record)
        return records

    def _append_event(
        self,
        event: JournalEvent,
        lifecycle: JournalLifecycle,
        *,
        state_hash_before: str | None = None,
        state_hash_after: str | None = None,
        snapshot_hash: str | None = None,
        snapshot_sequence: int | None = None,
        failure_category: str | None = None,
    ) -> JournalRecord:
        _reject_private_payload(getattr(event, "payload", {}))
        event_type = getattr(event.event_type, "value", str(event.event_type))
        return self._append(
            lifecycle=lifecycle,
            event_id=_safe_label(event.event_id),
            event_type=_safe_label(str(event_type)),
            source=_safe_label(event.source),
            processing_sequence=event.processing_sequence,
            causation_id=_safe_optional_label(event.causation_id),
            correlation_id=_safe_optional_label(event.correlation_id),
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            snapshot_hash=snapshot_hash,
            snapshot_sequence=snapshot_sequence,
            failure_category=failure_category,
        )

    def _append(self, **values: Any) -> JournalRecord:
        started = time.perf_counter()
        status = "failure"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                record = _new_record(previous_record_hash=self._last_hash, **values)
                line = record.model_dump_json() + "\n"
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(line)
                    output.flush()
                    os.fsync(output.fileno())
                _fsync_directory(self.path.parent)
                self._last_hash = record.record_hash
                if record.processing_sequence is not None:
                    self._last_sequence = max(
                        self._last_sequence, record.processing_sequence
                    )
                if (
                    record.snapshot_hash is not None
                    and record.snapshot_sequence is not None
                ):
                    self._last_snapshot_hash = record.snapshot_hash
                    self._last_snapshot_sequence = record.snapshot_sequence
                status = "success"
                return record
        finally:
            if self._telemetry is not None:
                try:
                    self._telemetry.storage_observation(
                        "journal", "append", status, time.perf_counter() - started
                    )
                except Exception:
                    pass

    def _rotate_if_needed(self) -> None:
        if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.retained_files}")
        oldest.unlink(missing_ok=True)
        for index in range(self.retained_files - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                os.replace(source, self.path.with_name(f"{self.path.name}.{index + 1}"))
        os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        checkpoint = _new_record(
            previous_record_hash=self._last_hash,
            lifecycle=JournalLifecycle.CHECKPOINT,
            event_id="journal-rotation",
            event_type="state_snapshot",
            source="journal.rotation",
            processing_sequence=self._last_sequence,
            snapshot_sequence=self._last_snapshot_sequence,
            snapshot_hash=self._last_snapshot_hash,
        )
        with self.path.open("x", encoding="utf-8") as output:
            output.write(checkpoint.model_dump_json() + "\n")
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(self.path.parent)
        self._last_hash = checkpoint.record_hash

    def _checkpoint(self, snapshot: AgentStateSnapshot) -> JournalRecord:
        return self._append(
            lifecycle=JournalLifecycle.CHECKPOINT,
            event_id="journal-bootstrap",
            event_type="state_snapshot",
            source="journal.bootstrap",
            processing_sequence=snapshot.last_processed_event_sequence,
            snapshot_sequence=snapshot.last_processed_event_sequence,
            snapshot_hash=hash_snapshot(snapshot),
        )

    def _classify_interrupted(
        self,
        record: JournalRecord,
        snapshot: AgentStateSnapshot,
        snapshot_hash: str,
    ) -> JournalRecord:
        sequence = record.processing_sequence
        if sequence is None:
            category = "started_without_sequence"
        elif snapshot.last_processed_event_sequence < sequence:
            category = "uncommitted_after_crash"
        elif (
            snapshot.last_processed_event_sequence == sequence
            and record.lifecycle == JournalLifecycle.PREPARED
            and record.state_hash_after == snapshot_hash
        ):
            category = "committed_before_crash"
        else:
            raise JournalIntegrityError("interrupted event cannot be reconciled")
        return self._recovery_record(
            record,
            category,
            snapshot_hash,
            snapshot.last_processed_event_sequence,
        )

    def _recovery_record(
        self,
        record: JournalRecord,
        category: str,
        snapshot_hash: str,
        snapshot_sequence: int,
    ) -> JournalRecord:
        return self._append(
            lifecycle=JournalLifecycle.RECOVERY_CLASSIFIED,
            event_id=record.event_id,
            event_type=record.event_type,
            source="journal.recovery",
            processing_sequence=record.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            causation_id=record.causation_id,
            correlation_id=record.correlation_id,
            snapshot_hash=snapshot_hash,
            failure_category=category,
        )

    def _journal_paths(self) -> list[Path]:
        indexed = [
            (
                index,
                self.path.with_name(f"{self.path.name}.{index}"),
            )
            for index in range(1, self.retained_files + 1)
        ]
        existing = [(index, path) for index, path in indexed if path.exists()]
        if existing:
            indices = [index for index, _path in existing]
            if indices != list(range(1, max(indices) + 1)):
                raise JournalIntegrityError(
                    "rotated journal file sequence is incomplete"
                )
            if not self.path.exists():
                raise JournalIntegrityError("active journal file is missing")
        return [path for _index, path in reversed(existing)] + (
            [self.path] if self.path.exists() else []
        )


def _reject_private_payload(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(
                character for character in str(key).lower() if character.isalnum()
            )
            if normalized == "hiddenthought":
                raise JournalIntegrityError(
                    "hidden thought is forbidden in the event journal"
                )
            _reject_private_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private_payload(item)


def hash_snapshot(snapshot: AgentStateSnapshot) -> str:
    canonical = json.dumps(
        snapshot.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _new_record(*, previous_record_hash: str | None, **values: Any) -> JournalRecord:
    payload = {
        "schema_version": 2,
        "record_id": str(uuid4()),
        "timestamp": datetime.now(UTC),
        "previous_record_hash": previous_record_hash,
        **values,
    }
    provisional = JournalRecord(record_hash="0" * 64, **payload)
    return provisional.model_copy(update={"record_hash": _record_hash(provisional)})


def _record_hash(record: JournalRecord) -> str:
    exclude = {"record_hash"}
    if record.schema_version == 1:
        exclude.update({"actor_id", "actor_role", "target", "reauthenticated"})
    payload = record.model_dump(mode="json", exclude=exclude)
    canonical = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _safe_optional_label(value: str | None) -> str | None:
    return None if value is None else _safe_label(value)


def _safe_label(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9._:@-]{1,128}", value) else "redacted"


def _safe_target(value: str) -> str:
    return (
        value
        if re.fullmatch(r"[A-Z]+ /[A-Za-z0-9._:@/*-]{1,240}", value)
        else "redacted"
    )


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

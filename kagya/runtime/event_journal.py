"""Durable, integrity-chained lifecycle evidence for agent events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from kagya.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType


CURRENT_EVENT_JOURNAL_SCHEMA_VERSION: Literal[1] = 1
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN = b"PROJECT-KAGYA:event-journal:v1\0"


class EventLifecycle(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    PREPARED = "prepared"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERY_CLASSIFIED = "recovery_classified"
    CHECKPOINT = "checkpoint"


class EventFailureCategory(str, Enum):
    HANDLER_FAILURE = "handler_failure"
    ACCEPTED_NOT_STARTED = "accepted_not_started"
    UNCOMMITTED_AFTER_CRASH = "uncommitted_after_crash"
    COMMITTED_BEFORE_CRASH = "committed_before_crash"


class EventJournalAppendStage(str, Enum):
    VALIDATE = "validate"
    WRITE = "write"
    FILE_FSYNC = "file_fsync"
    PARENT_FSYNC = "parent_fsync"
    ROTATION = "rotation"


class _JournalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EventJournalRecord(_JournalModel):
    schema_version: Literal[1] = CURRENT_EVENT_JOURNAL_SCHEMA_VERSION
    record_id: str = Field(min_length=1)
    timestamp: datetime
    lifecycle: EventLifecycle
    event_id: str | None = None
    event_type: AgentEventType | None = None
    source: AgentEventSource | None = None
    processing_sequence: int | None = Field(default=None, ge=0)
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    snapshot_sequence: int | None = Field(default=None, ge=0)
    snapshot_hash: str | None = None
    failure_category: EventFailureCategory | None = None
    previous_record_hash: str | None = None
    record_hash: str

    @field_validator("record_id", "event_id")
    @classmethod
    def require_canonical_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = UUID(value)
        except ValueError:
            raise ValueError("identifier must be a UUID") from None
        if str(parsed) != value:
            raise ValueError("identifier must use canonical UUID form")
        return value

    @model_validator(mode="after")
    def validate_record_shape(self) -> EventJournalRecord:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        for value in (
            self.state_hash_before,
            self.state_hash_after,
            self.snapshot_hash,
            self.previous_record_hash,
            self.record_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("hash must be lowercase SHA-256")

        identity = (self.event_id, self.event_type, self.source)
        has_identity = all(value is not None for value in identity)
        no_identity = all(value is None for value in identity)
        if not has_identity and not no_identity:
            raise ValueError("event identity must be complete")

        if self.lifecycle is EventLifecycle.CHECKPOINT:
            if not no_identity or self.processing_sequence is None:
                raise ValueError("checkpoint identity is invalid")
            self._require_snapshot()
            self._forbid_state_and_failure()
        elif not has_identity:
            raise ValueError("event lifecycle requires identity")
        elif self.lifecycle is EventLifecycle.ACCEPTED:
            self._require_only()
        elif self.lifecycle is EventLifecycle.STARTED:
            if self.processing_sequence is None:
                raise ValueError("started requires processing sequence")
            self._forbid_state_snapshot_failure()
        elif self.lifecycle is EventLifecycle.PREPARED:
            if (
                self.processing_sequence is None
                or self.state_hash_before is None
                or self.state_hash_after is None
            ):
                raise ValueError("prepared requires sequence and state hashes")
            if any(
                value is not None
                for value in (
                    self.snapshot_sequence,
                    self.snapshot_hash,
                    self.failure_category,
                )
            ):
                raise ValueError("prepared has forbidden fields")
        elif self.lifecycle is EventLifecycle.COMPLETED:
            if self.processing_sequence is None:
                raise ValueError("completed requires processing sequence")
            self._require_snapshot()
            self._forbid_state_and_failure()
        elif self.lifecycle is EventLifecycle.FAILED:
            if self.processing_sequence is None:
                raise ValueError("failed requires processing sequence")
            self._require_snapshot()
            if self.failure_category is not EventFailureCategory.HANDLER_FAILURE:
                raise ValueError("failed category is invalid")
            self._forbid_state()
        elif self.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED:
            self._require_snapshot()
            self._forbid_state()
            if self.failure_category is EventFailureCategory.ACCEPTED_NOT_STARTED:
                if self.processing_sequence is not None:
                    raise ValueError("accepted-only recovery cannot consume sequence")
            elif self.failure_category not in {
                EventFailureCategory.UNCOMMITTED_AFTER_CRASH,
                EventFailureCategory.COMMITTED_BEFORE_CRASH,
            }:
                raise ValueError("recovery category is invalid")
            elif self.processing_sequence is None:
                raise ValueError("processing recovery requires sequence")
        return self

    def _require_only(self) -> None:
        if any(
            value is not None
            for value in (
                self.processing_sequence,
                self.state_hash_before,
                self.state_hash_after,
                self.snapshot_sequence,
                self.snapshot_hash,
                self.failure_category,
            )
        ):
            raise ValueError("accepted has forbidden fields")

    def _require_snapshot(self) -> None:
        if self.snapshot_sequence is None or self.snapshot_hash is None:
            raise ValueError("snapshot identity is required")

    def _forbid_state(self) -> None:
        if self.state_hash_before is not None or self.state_hash_after is not None:
            raise ValueError("state hashes are forbidden")

    def _forbid_state_and_failure(self) -> None:
        self._forbid_state()
        if self.failure_category is not None:
            raise ValueError("failure category is forbidden")

    def _forbid_state_snapshot_failure(self) -> None:
        self._forbid_state_and_failure()
        if self.snapshot_sequence is not None or self.snapshot_hash is not None:
            raise ValueError("snapshot identity is forbidden")


class EventJournalRecovery(_JournalModel):
    processing_high_water: int = Field(ge=0)
    snapshot_sequence: int = Field(ge=0)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class EventJournalError(Exception):
    """Base class for bounded Journal failures."""


class EventJournalLoadError(EventJournalError):
    """Journal artifacts cannot be loaded safely."""


class EventJournalIntegrityError(EventJournalLoadError):
    """Journal evidence is internally inconsistent."""


class UnsupportedEventJournalVersion(EventJournalLoadError):
    """A record uses an unsupported schema version."""


class EventJournalAppendError(EventJournalError):
    """A lifecycle record did not reach confirmed durable success."""

    def __init__(self, stage: EventJournalAppendStage, *, published: bool) -> None:
        self.stage = stage
        self.published = published
        super().__init__(
            "EventJournal append failed "
            f"at {stage.value}; published={str(published).lower()}"
        )


@dataclass(frozen=True, slots=True)
class _EventState:
    event_id: str
    event_type: AgentEventType
    source: AgentEventSource
    requested_at: datetime
    lifecycle: EventLifecycle
    processing_sequence: int | None
    state_hash_before: str | None = None
    state_hash_after: str | None = None


@dataclass(frozen=True, slots=True)
class _VerifiedJournal:
    records: tuple[EventJournalRecord, ...]
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    open_events: tuple[_EventState, ...]


class EventJournalLease:
    """Exclusive process lease protecting Journal and Snapshot startup authority."""

    def __init__(self, journal_path: str | Path) -> None:
        self.path = Path(journal_path)
        self._descriptor: int | None = None
        lease_failure: EventJournalLoadError | None = None
        descriptor: int | None = None
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_status = self.path.parent.lstat()
            if (
                not stat.S_ISDIR(parent_status.st_mode)
                or parent_status.st_uid != os.geteuid()
                or parent_status.st_mode & 0o077
            ):
                raise OSError("journal directory is not private")
            lock_path = self.path.with_name(f".{self.path.name}.lock")
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("journal lock is not regular")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._descriptor = descriptor
            descriptor = None
            self._fsync_parent()
        except OSError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            lease_failure = EventJournalLoadError(
                "EventJournal exclusive authority is unavailable"
            )
        if lease_failure is not None:
            raise lease_failure

    @property
    def held(self) -> bool:
        return self._descriptor is not None

    def close(self) -> None:
        if self._descriptor is not None:
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)
                self._descriptor = None

    def _transfer(self) -> EventJournalLease:
        if self._descriptor is None:
            raise ValueError("EventJournal lease is not held")
        adopted = object.__new__(EventJournalLease)
        adopted.path = self.path
        adopted._descriptor = self._descriptor
        self._descriptor = None
        return adopted

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _fsync_parent(self) -> None:
        descriptor = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class EventJournal:
    """Append and verify durable lifecycle records without event payloads."""

    def __init__(
        self,
        path: str | Path,
        max_bytes: int,
        retained_files: int,
        *,
        clock: Callable[[], datetime] | None = None,
        append_stage_hook: (
            Callable[[EventLifecycle, EventJournalAppendStage], None] | None
        ) = None,
        lease: EventJournalLease | None = None,
    ) -> None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(retained_files, bool)
            or not isinstance(retained_files, int)
            or retained_files < 2
        ):
            raise ValueError("retained_files must be at least two")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.retained_files = retained_files
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._append_stage_hook = append_stage_hook
        self._lock = RLock()
        if lease is not None and (lease.path != self.path or not lease.held):
            raise ValueError("EventJournal lease does not match configured path")
        self._lease = (
            lease._transfer() if lease is not None else EventJournalLease(self.path)
        )
        with self._lock:
            try:
                records = self._read_records_unlocked()
                if records:
                    self._verify_records(records)
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Release this process's exclusive journal authority."""

        with self._lock:
            self._lease.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def records(self) -> tuple[EventJournalRecord, ...]:
        with self._lock:
            self._require_authority()
            return self._read_records_unlocked()

    @property
    def has_exclusive_authority(self) -> bool:
        return self._lease.held

    def append_accepted(self, event: AgentEvent) -> None:
        self._append_event(EventLifecycle.ACCEPTED, event)

    def append_started(self, event: AgentEvent) -> None:
        self._append_event(
            EventLifecycle.STARTED,
            event,
            processing_sequence=event.processing_sequence,
        )

    def append_prepared(
        self,
        event: AgentEvent,
        state_hash_before: str,
        state_hash_after: str,
    ) -> None:
        self._append_event(
            EventLifecycle.PREPARED,
            event,
            processing_sequence=event.processing_sequence,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
        )

    def append_completed(
        self,
        event: AgentEvent,
        snapshot_sequence: int,
        snapshot_hash: str,
    ) -> None:
        self._append_event(
            EventLifecycle.COMPLETED,
            event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
        )

    def append_failed(
        self,
        event: AgentEvent,
        snapshot_sequence: int,
        snapshot_hash: str,
    ) -> None:
        self._append_event(
            EventLifecycle.FAILED,
            event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            failure_category=EventFailureCategory.HANDLER_FAILURE,
        )

    def verify_and_reconcile(
        self, snapshot_sequence: int, snapshot_hash: str
    ) -> EventJournalRecovery:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                checkpoint = self._make_record(
                    EventLifecycle.CHECKPOINT,
                    previous_hash=None,
                    processing_sequence=snapshot_sequence,
                    snapshot_sequence=snapshot_sequence,
                    snapshot_hash=snapshot_hash,
                )
                self._append_record_unlocked(checkpoint)
                return EventJournalRecovery(
                    processing_high_water=snapshot_sequence,
                    snapshot_sequence=snapshot_sequence,
                    snapshot_hash=snapshot_hash,
                )

            verified = self._verify_records(records)
            processing = tuple(
                event
                for event in verified.open_events
                if event.lifecycle in {EventLifecycle.STARTED, EventLifecycle.PREPARED}
            )
            if len(processing) > 1:
                raise EventJournalIntegrityError(
                    "EventJournal has multiple interrupted handlers"
                )

            canonical_matches = (
                snapshot_sequence == verified.snapshot_sequence
                and snapshot_hash == verified.snapshot_hash
            )
            if processing:
                interrupted = processing[0]
                category = EventFailureCategory.UNCOMMITTED_AFTER_CRASH
                if interrupted.lifecycle is EventLifecycle.PREPARED:
                    if (
                        snapshot_sequence == interrupted.processing_sequence
                        and snapshot_hash == interrupted.state_hash_after
                    ):
                        category = EventFailureCategory.COMMITTED_BEFORE_CRASH
                    elif not (
                        canonical_matches
                        and snapshot_hash == interrupted.state_hash_before
                    ):
                        raise EventJournalIntegrityError(
                            "Prepared transition does not match canonical snapshot"
                        )
                elif not canonical_matches:
                    raise EventJournalIntegrityError(
                        "Started event does not match canonical snapshot"
                    )
                self._append_recovery_unlocked(
                    interrupted,
                    snapshot_sequence,
                    snapshot_hash,
                    category,
                )
                records = self._read_records_unlocked()
                verified = self._verify_records(records)
            elif not canonical_matches:
                raise EventJournalIntegrityError(
                    "Journal and canonical snapshot are inconsistent"
                )

            for accepted in tuple(
                event
                for event in verified.open_events
                if event.lifecycle is EventLifecycle.ACCEPTED
            ):
                self._append_recovery_unlocked(
                    accepted,
                    snapshot_sequence,
                    snapshot_hash,
                    EventFailureCategory.ACCEPTED_NOT_STARTED,
                )
                verified = self._verify_records(self._read_records_unlocked())

            return EventJournalRecovery(
                processing_high_water=verified.processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
            )

    def _append_event(
        self,
        lifecycle: EventLifecycle,
        event: AgentEvent,
        **fields: object,
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                raise EventJournalIntegrityError(
                    "EventJournal requires startup reconciliation"
                )
            validation_failure: EventJournalAppendError | None = None
            try:
                record = self._make_record(
                    lifecycle,
                    previous_hash=records[-1].record_hash,
                    event=event,
                    **fields,
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                validation_failure = EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                )
            if validation_failure is not None:
                raise validation_failure
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def _append_recovery_unlocked(
        self,
        event: _EventState,
        snapshot_sequence: int,
        snapshot_hash: str,
        category: EventFailureCategory,
    ) -> None:
        records = self._read_records_unlocked()
        journal_event = AgentEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            source=event.source,
            requested_at=event.requested_at,
            processing_sequence=event.processing_sequence,
        )
        record = self._make_record(
            EventLifecycle.RECOVERY_CLASSIFIED,
            previous_hash=records[-1].record_hash,
            event=journal_event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            failure_category=category,
        )
        self._verify_records((*records, record))
        self._append_record_unlocked(record)
        self._maybe_rotate_unlocked()

    def _make_record(
        self,
        lifecycle: EventLifecycle,
        *,
        previous_hash: str | None,
        event: AgentEvent | None = None,
        **fields: object,
    ) -> EventJournalRecord:
        validation_failure: EventJournalAppendError | None = None
        try:
            timestamp = self._clock()
            unsigned = EventJournalRecord.model_validate(
                {
                    "record_id": str(uuid4()),
                    "timestamp": timestamp,
                    "lifecycle": lifecycle,
                    "event_id": event.event_id if event is not None else None,
                    "event_type": event.event_type if event is not None else None,
                    "source": event.source if event is not None else None,
                    "previous_record_hash": previous_hash,
                    "record_hash": "0" * 64,
                    **fields,
                }
            )
            record = unsigned.model_copy(
                update={"record_hash": self._record_hash(unsigned)}
            )
        except Exception:
            validation_failure = EventJournalAppendError(
                EventJournalAppendStage.VALIDATE, published=False
            )
        if validation_failure is not None:
            raise validation_failure
        return record

    @staticmethod
    def _record_hash(record: EventJournalRecord) -> str:
        canonical = json.dumps(
            record.model_dump(mode="json", exclude={"record_hash"}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(_HASH_DOMAIN + canonical).hexdigest()

    @staticmethod
    def _record_bytes(record: EventJournalRecord) -> bytes:
        return (
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def _append_record_unlocked(self, record: EventJournalRecord) -> None:
        stage = EventJournalAppendStage.WRITE
        published = False
        append_failure: EventJournalAppendError | None = None
        descriptor: int | None = None
        try:
            self._run_hook(record.lifecycle, stage)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("journal is not a regular file")
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab") as journal_file:
                descriptor = None
                payload = self._record_bytes(record)
                if journal_file.write(payload) != len(payload):
                    raise OSError("incomplete journal write")
                journal_file.flush()
                stage = EventJournalAppendStage.FILE_FSYNC
                self._run_hook(record.lifecycle, stage)
                os.fsync(journal_file.fileno())
                published = True
            stage = EventJournalAppendStage.PARENT_FSYNC
            self._run_hook(record.lifecycle, stage)
            self._fsync_parent()
        except Exception:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            append_failure = EventJournalAppendError(stage, published=published)
        if append_failure is not None:
            raise append_failure

    def _maybe_rotate_unlocked(self) -> None:
        failure: EventJournalAppendError | None
        try:
            size = self.path.stat().st_size
        except OSError:
            failure = EventJournalAppendError(
                EventJournalAppendStage.ROTATION, published=True
            )
        else:
            failure = None
        if failure is not None:
            raise failure
        if size <= self.max_bytes:
            return
        verified = self._verify_records(self._read_records_unlocked())
        if verified.open_events:
            return
        self._rotate_unlocked(verified)

    def _rotate_unlocked(self, verified: _VerifiedJournal) -> None:
        failure: EventJournalAppendError | None = None
        try:
            rotated = self._rotated_paths_unlocked()
            next_index = rotated[-1][0] + 1 if rotated else 0
            archived = self._segment_path(next_index)
            os.replace(self.path, archived)
            self._fsync_parent()
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=verified.records[-1].record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
            )
            temporary = self.path.with_name(f".{self.path.name}.rotation.tmp")
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as segment_file:
                segment_file.write(self._record_bytes(checkpoint))
                segment_file.flush()
                os.fsync(segment_file.fileno())
            os.replace(temporary, self.path)
            self._fsync_parent()
            rotated = self._rotated_paths_unlocked()
            while len(rotated) + 1 > self.retained_files:
                rotated[0][1].unlink()
                self._fsync_parent()
                rotated = rotated[1:]
        except Exception:
            failure = EventJournalAppendError(
                EventJournalAppendStage.ROTATION, published=True
            )
        if failure is not None:
            raise failure

    def _read_records_unlocked(self) -> tuple[EventJournalRecord, ...]:
        artifacts = self._journal_artifacts_unlocked()
        if not artifacts:
            return ()
        records: list[EventJournalRecord] = []
        previous_tail: str | None = None
        for segment_index, path in artifacts:
            segment = self._read_segment(path)
            if not segment:
                raise EventJournalLoadError("EventJournal segment is empty")
            if segment[0].lifecycle is not EventLifecycle.CHECKPOINT:
                raise EventJournalIntegrityError(
                    "EventJournal segment lacks checkpoint anchor"
                )
            if (
                previous_tail is not None
                and segment[0].previous_record_hash != previous_tail
            ):
                raise EventJournalIntegrityError("EventJournal segment chain is broken")
            if segment_index is not None and path == self.path:
                raise EventJournalIntegrityError(
                    "EventJournal segment identity is invalid"
                )
            for index, record in enumerate(segment):
                if self._record_hash(record) != record.record_hash:
                    raise EventJournalIntegrityError(
                        "EventJournal record hash mismatch"
                    )
                if (
                    index > 0
                    and record.previous_record_hash != segment[index - 1].record_hash
                ):
                    raise EventJournalIntegrityError(
                        "EventJournal record chain is broken"
                    )
            records.extend(segment)
            previous_tail = segment[-1].record_hash
        return tuple(records)

    def _read_segment(self, path: Path) -> tuple[EventJournalRecord, ...]:
        read_failure: EventJournalLoadError | None = None
        descriptor: int | None = None
        try:
            status = path.lstat()
            if not stat.S_ISREG(status.st_mode):
                raise OSError("journal segment is not regular")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("journal segment is not regular")
            with os.fdopen(descriptor, "rb") as segment_file:
                descriptor = None
                raw = segment_file.read()
        except OSError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            read_failure = EventJournalLoadError("EventJournal segment cannot be read")
        if read_failure is not None:
            raise read_failure
        if not raw or not raw.endswith(b"\n"):
            raise EventJournalLoadError("EventJournal contains a partial record")

        records: list[EventJournalRecord] = []
        parse_failure: EventJournalLoadError | None = None
        for line in raw.splitlines():
            try:
                value = json.loads(
                    line.decode("utf-8"),
                    parse_constant=self._reject_json_constant,
                )
                if not isinstance(value, dict):
                    raise ValueError("record root is invalid")
                version = value.get("schema_version")
                if (
                    isinstance(version, int)
                    and not isinstance(version, bool)
                    and version != 1
                ):
                    raise UnsupportedEventJournalVersion(
                        "EventJournal schema version is unsupported"
                    )
                records.append(EventJournalRecord.model_validate_json(line))
            except UnsupportedEventJournalVersion:
                raise
            except (UnicodeError, json.JSONDecodeError, ValueError, ValidationError):
                parse_failure = EventJournalLoadError(
                    "EventJournal contains an invalid record"
                )
                break
        if parse_failure is not None:
            raise parse_failure
        return tuple(records)

    def _journal_artifacts_unlocked(self) -> list[tuple[int | None, Path]]:
        scan_failure: EventJournalLoadError | None = None
        try:
            candidates = tuple(self.path.parent.glob(f"{self.path.name}.*"))
        except OSError:
            candidates = ()
            scan_failure = EventJournalLoadError(
                "EventJournal artifacts cannot be inspected"
            )
        if scan_failure is not None:
            raise scan_failure

        rotated: list[tuple[int, Path]] = []
        interrupted_failure: EventJournalLoadError | None = None
        try:
            interrupted = tuple(self.path.parent.glob(f".{self.path.name}.rotation*"))
        except OSError:
            interrupted = ()
            interrupted_failure = EventJournalLoadError(
                "EventJournal artifacts cannot be inspected"
            )
        if interrupted_failure is not None:
            raise interrupted_failure
        if interrupted:
            raise EventJournalIntegrityError(
                "EventJournal has an interrupted rotation artifact"
            )
        for candidate in candidates:
            suffix = candidate.name.removeprefix(f"{self.path.name}.")
            if not re.fullmatch(r"\d{8}", suffix):
                raise EventJournalIntegrityError(
                    "EventJournal has an interrupted rotation artifact"
                )
            rotated.append((int(suffix), candidate))
        rotated.sort()
        indexes = [index for index, _ in rotated]
        if indexes and indexes != list(range(indexes[0], indexes[-1] + 1)):
            raise EventJournalIntegrityError("EventJournal rotated segment is missing")

        active_failure: EventJournalLoadError | None = None
        try:
            active_exists = self.path.lstat() is not None
        except FileNotFoundError:
            active_exists = False
        except OSError:
            active_exists = False
            active_failure = EventJournalLoadError("EventJournal cannot be inspected")
        if active_failure is not None:
            raise active_failure
        if rotated and not active_exists:
            raise EventJournalIntegrityError("EventJournal active segment is missing")
        artifacts: list[tuple[int | None, Path]] = [*rotated]
        if active_exists:
            artifacts.append((None, self.path))
        return artifacts

    def _verify_records(
        self, records: tuple[EventJournalRecord, ...]
    ) -> _VerifiedJournal:
        if not records or records[0].lifecycle is not EventLifecycle.CHECKPOINT:
            raise EventJournalIntegrityError("EventJournal lacks checkpoint authority")
        checkpoint = records[0]
        assert checkpoint.processing_sequence is not None
        assert checkpoint.snapshot_sequence is not None
        assert checkpoint.snapshot_hash is not None
        high_water = checkpoint.processing_sequence
        snapshot_sequence = checkpoint.snapshot_sequence
        snapshot_hash = checkpoint.snapshot_hash
        if snapshot_sequence > high_water:
            raise EventJournalIntegrityError("Checkpoint sequence is impossible")
        open_events: dict[str, _EventState] = {}
        seen_event_ids: set[str] = set()
        seen_record_ids: set[str] = set()

        for record in records:
            if record.record_id in seen_record_ids:
                raise EventJournalIntegrityError("Record identifier is duplicated")
            seen_record_ids.add(record.record_id)

        for record in records[1:]:
            if record.lifecycle is EventLifecycle.CHECKPOINT:
                if open_events:
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open event lifecycle"
                    )
                if (
                    record.processing_sequence != high_water
                    or record.snapshot_sequence != snapshot_sequence
                    or record.snapshot_hash != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Checkpoint continuity is invalid")
                continue
            assert record.event_id is not None
            assert record.event_type is not None
            assert record.source is not None
            current = open_events.get(record.event_id)
            if record.lifecycle is EventLifecycle.ACCEPTED:
                if record.event_id in seen_event_ids:
                    raise EventJournalIntegrityError("Event identifier is duplicated")
                seen_event_ids.add(record.event_id)
                open_events[record.event_id] = _EventState(
                    record.event_id,
                    record.event_type,
                    record.source,
                    record.timestamp,
                    EventLifecycle.ACCEPTED,
                    None,
                )
                continue
            if current is None:
                raise EventJournalIntegrityError("Event lifecycle lacks acceptance")
            if (record.event_type, record.source) != (
                current.event_type,
                current.source,
            ):
                raise EventJournalIntegrityError("Event metadata changed")

            if record.lifecycle is EventLifecycle.STARTED:
                if current.lifecycle is not EventLifecycle.ACCEPTED:
                    raise EventJournalIntegrityError("Started lifecycle is impossible")
                if record.processing_sequence != high_water + 1:
                    raise EventJournalIntegrityError("Processing sequence has a gap")
                high_water = record.processing_sequence
                open_events[record.event_id] = _EventState(
                    current.event_id,
                    current.event_type,
                    current.source,
                    current.requested_at,
                    EventLifecycle.STARTED,
                    record.processing_sequence,
                )
            elif record.lifecycle is EventLifecycle.PREPARED:
                if (
                    current.lifecycle is not EventLifecycle.STARTED
                    or record.processing_sequence != current.processing_sequence
                    or record.state_hash_before != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Prepared lifecycle is impossible")
                open_events[record.event_id] = _EventState(
                    current.event_id,
                    current.event_type,
                    current.source,
                    current.requested_at,
                    EventLifecycle.PREPARED,
                    current.processing_sequence,
                    record.state_hash_before,
                    record.state_hash_after,
                )
            elif record.lifecycle is EventLifecycle.COMPLETED:
                if (
                    current.lifecycle is not EventLifecycle.PREPARED
                    or record.processing_sequence != current.processing_sequence
                    or record.snapshot_sequence != current.processing_sequence
                    or record.snapshot_hash != current.state_hash_after
                ):
                    raise EventJournalIntegrityError(
                        "Completed lifecycle is impossible"
                    )
                assert record.snapshot_sequence is not None
                assert record.snapshot_hash is not None
                snapshot_sequence = record.snapshot_sequence
                snapshot_hash = record.snapshot_hash
                del open_events[record.event_id]
            elif record.lifecycle is EventLifecycle.FAILED:
                if (
                    current.lifecycle is not EventLifecycle.STARTED
                    or record.processing_sequence != current.processing_sequence
                    or record.snapshot_sequence != snapshot_sequence
                    or record.snapshot_hash != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Failed lifecycle is impossible")
                del open_events[record.event_id]
            elif record.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED:
                if record.snapshot_sequence is None or record.snapshot_hash is None:
                    raise EventJournalIntegrityError("Recovery snapshot is missing")
                if record.failure_category is EventFailureCategory.ACCEPTED_NOT_STARTED:
                    if current.lifecycle is not EventLifecycle.ACCEPTED:
                        raise EventJournalIntegrityError(
                            "Accepted recovery is impossible"
                        )
                elif (
                    record.failure_category
                    is EventFailureCategory.UNCOMMITTED_AFTER_CRASH
                ):
                    if (
                        current.lifecycle
                        not in {
                            EventLifecycle.STARTED,
                            EventLifecycle.PREPARED,
                        }
                        or record.processing_sequence != current.processing_sequence
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "Uncommitted recovery is impossible"
                        )
                elif (
                    record.failure_category
                    is EventFailureCategory.COMMITTED_BEFORE_CRASH
                ):
                    if (
                        current.lifecycle is not EventLifecycle.PREPARED
                        or record.processing_sequence != current.processing_sequence
                        or record.snapshot_sequence != current.processing_sequence
                        or record.snapshot_hash != current.state_hash_after
                    ):
                        raise EventJournalIntegrityError(
                            "Committed recovery is impossible"
                        )
                    snapshot_sequence = record.snapshot_sequence
                    snapshot_hash = record.snapshot_hash
                else:
                    raise EventJournalIntegrityError("Recovery category is impossible")
                del open_events[record.event_id]
            else:
                raise EventJournalIntegrityError("Lifecycle is impossible")

        return _VerifiedJournal(
            records,
            high_water,
            snapshot_sequence,
            snapshot_hash,
            tuple(open_events.values()),
        )

    def _rotated_paths_unlocked(self) -> list[tuple[int, Path]]:
        return [
            (index, path)
            for index, path in self._journal_artifacts_unlocked()
            if index is not None
        ]

    def _segment_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index:08d}")

    def _fsync_parent(self) -> None:
        descriptor = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _run_hook(
        self, lifecycle: EventLifecycle, stage: EventJournalAppendStage
    ) -> None:
        if self._append_stage_hook is not None:
            self._append_stage_hook(lifecycle, stage)

    def _require_authority(self) -> None:
        if not self._lease.held:
            raise EventJournalLoadError(
                "EventJournal exclusive authority is unavailable"
            )

    @staticmethod
    def _validate_snapshot_identity(sequence: int, snapshot_hash: str) -> None:
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or not isinstance(snapshot_hash, str)
            or _HASH_PATTERN.fullmatch(snapshot_hash) is None
        ):
            raise ValueError("canonical snapshot identity is invalid")

    @staticmethod
    def _reject_json_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

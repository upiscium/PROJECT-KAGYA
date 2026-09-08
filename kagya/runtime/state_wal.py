"""Durable, non-authoritative storage for reconstructible agent state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from kagya.privacy import normalize_private_key
from kagya.runtime.agent_state import AgentStateSnapshot, AgentStateStore


class StateWALError(Exception):
    """Bounded base class for WAL failures."""


class StateWALMissing(StateWALError):
    """The WAL has not been bootstrapped."""


class StateWALFormatError(StateWALError):
    """A WAL artifact is malformed or unsafe."""


class StateWALIntegrityError(StateWALError):
    """A WAL hash, lineage, or manifest constraint is invalid."""


class StateWALPermissionError(StateWALError):
    """A WAL artifact has unsafe permissions or ownership."""


class StateWALConflictError(StateWALError):
    """A requested operation conflicts with retained WAL state."""


class RecoveryReason(str, Enum):
    BOOTSTRAP = "bootstrap"
    EXACT_CURRENT_REPAIR = "exact_current_repair"
    TRUE_ROLLBACK = "true_rollback"
    UNCOMMITTED_TAIL = "uncommitted_tail"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BaselineRecord(_StrictModel):
    record_type: Literal["baseline"]
    schema_version: Literal[1]
    command_version: Literal[1]
    record_id: UUID
    generation_id: UUID
    created_at: datetime
    baseline_snapshot_sequence: int = Field(ge=0)
    baseline_snapshot_hash: str = Field(min_length=64, max_length=64)
    baseline_snapshot: AgentStateSnapshot
    journal_processing_high_water: int = Field(ge=0)
    predecessor_generation_id: UUID | None = None
    predecessor_generation_hash: str | None = None
    reason: RecoveryReason
    previous_record_hash: str | None
    record_hash: str = Field(min_length=64, max_length=64)


class TransitionRecord(_StrictModel):
    record_type: Literal["transition"]
    schema_version: Literal[1]
    command_version: Literal[1]
    record_id: UUID
    generation_id: UUID
    created_at: datetime
    event_id: UUID
    event_type: str = Field(min_length=1, max_length=128)
    event_source: str = Field(min_length=1, max_length=128)
    processing_sequence: int = Field(ge=0)
    prior_snapshot_sequence: int = Field(ge=0)
    prior_snapshot_hash: str = Field(min_length=64, max_length=64)
    candidate_snapshot_sequence: int = Field(ge=0)
    candidate_snapshot_hash: str = Field(min_length=64, max_length=64)
    candidate_snapshot: AgentStateSnapshot
    previous_record_hash: str = Field(min_length=64, max_length=64)
    record_hash: str = Field(min_length=64, max_length=64)


class Manifest(_StrictModel):
    schema_version: Literal[1]
    command_version: Literal[1]
    active_generation_id: UUID
    active_baseline_record_id: UUID
    active_baseline_record_hash: str = Field(min_length=64, max_length=64)
    predecessor_generation_id: UUID | None
    predecessor_generation_hash: str | None
    external_reconciliation_required: bool
    manifest_hash: str = Field(min_length=64, max_length=64)


class BootAnchor(_StrictModel):
    schema_version: Literal[1]
    command_version: Literal[1]
    snapshot_sequence: int = Field(ge=0)
    snapshot_hash: str = Field(min_length=64, max_length=64)
    generation_id: UUID
    anchored_record_id: UUID
    anchored_record_hash: str = Field(min_length=64, max_length=64)
    journal_processing_high_water: int = Field(ge=0)
    journal_tail_record_id: UUID | None
    journal_tail_record_hash: str | None
    journal_lineage_id: UUID
    anchor_hash: str = Field(min_length=64, max_length=64)


class StateWALInspection(_StrictModel):
    exists: bool
    active_manifest: Manifest | None
    records: tuple[BaselineRecord | TransitionRecord, ...]
    record_hashes: tuple[str, ...]
    baseline_record_id: UUID | None
    baseline_record_hash: str | None
    latest_snapshot_sequence: int | None
    latest_snapshot_hash: str | None


class StateWALFieldChange(_StrictModel):
    field: str
    before: Any
    after: Any


class DryRunDiff(_StrictModel):
    target_snapshot_sequence: int
    target_snapshot_hash: str
    current_snapshot_sequence: int
    current_snapshot_hash: str
    changes: tuple[StateWALFieldChange, ...]
    records_replayed: int
    external_side_effects_replayed: Literal[False]


_FORBIDDEN = frozenset(
    {
        "hiddenthought",
        "privatereasoning",
        "reasoning",
        "chainofthought",
        "prompt",
        "rawprompt",
        "systemprompt",
        "userprompt",
        "assistantprompt",
        "retrievedmemory",
        "privatestate",
        "turns",
        "attachments",
        "eventpayload",
        "requestpayload",
        "debugtrace",
        "transcript",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _reject_private(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if normalize_private_key(str(key)) in _FORBIDDEN:
                raise StateWALFormatError("WAL data violates privacy policy")
            _reject_private(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private(child)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _domain_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode() + b"\0" + _canonical(value)).hexdigest()


def _record_json(model: BaseModel, domain: str) -> tuple[dict[str, Any], str]:
    raw = model.model_dump(mode="json")
    unsigned = {key: value for key, value in raw.items() if key != "record_hash"}
    digest = _domain_hash(domain, unsigned)
    raw["record_hash"] = digest
    return raw, digest


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StateWAL:
    """A durable WAL that has no authority to publish canonical state."""

    def __init__(
        self,
        directory: str | Path,
        *,
        canonical_bytes: Callable[[AgentStateSnapshot], bytes] | None = None,
        snapshot_hash: Callable[[AgentStateSnapshot], str] | None = None,
        failure_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(directory)
        self._canonical_bytes = canonical_bytes or AgentStateStore._canonical_bytes
        self._snapshot_hash = snapshot_hash
        self._failure_hook = failure_hook
        self._lock = RLock()

    def _stage(self, stage: str) -> None:
        if self._failure_hook is not None:
            self._failure_hook(stage)

    @staticmethod
    def _check_dir(path: Path, *, create: bool) -> None:
        descriptor = -1
        try:
            descriptor = (
                StateWAL._open_or_create_private_dir(path)
                if create
                else StateWAL._open_private_dir(path, require_private=False)
            )
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
                raise OSError
            if stat.S_IMODE(status.st_mode) != 0o700:
                os.fchmod(descriptor, 0o700)
                os.fsync(descriptor)
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
                    raise OSError
        except Exception:
            raise StateWALPermissionError("WAL directory is unsafe") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _validate_descriptor(descriptor: int, *, regular: bool = True) -> None:
        status = os.fstat(descriptor)
        if (
            (regular and not stat.S_ISREG(status.st_mode))
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o600
        ):
            raise OSError

    @staticmethod
    def _open_private_dir(path: Path, *, require_private: bool = True) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        absolute = path.absolute()
        descriptor = os.open(absolute.anchor, flags)
        try:
            for component in absolute.parts[1:]:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or (require_private and stat.S_IMODE(status.st_mode) != 0o700)
            ):
                raise OSError
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_or_create_private_dir(path: Path) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        absolute = path.absolute()
        descriptor = os.open(absolute.anchor, flags)
        try:
            for component in absolute.parts[1:]:
                try:
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _prepare_for_write(self) -> None:
        self._check_dir(self.root.parent, create=True)
        self._check_dir(self.root, create=True)
        self._check_dir(self.root / "generations", create=True)

    def _prepare_for_read(self) -> bool:
        try:
            self._check_dir(self.root.parent, create=False)
            self._check_dir(self.root, create=False)
            self._check_dir(self.root / "generations", create=False)
        except StateWALPermissionError:
            if not self.root.exists():
                return False
            raise
        return True

    @staticmethod
    def _read_regular(path: Path) -> bytes:
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = StateWAL._open_private_dir(path.parent)
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            StateWAL._validate_descriptor(descriptor)
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                return source.read()
        except FileNotFoundError:
            raise StateWALMissing("WAL artifact is absent") from None
        except Exception:
            raise StateWALFormatError("WAL artifact is invalid") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        descriptor = StateWAL._open_private_dir(path, require_private=False)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(path.parent)
            try:
                existing = os.stat(
                    path.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    not stat.S_ISREG(existing.st_mode)
                    or existing.st_uid != os.geteuid()
                    or stat.S_IMODE(existing.st_mode) != 0o600
                ):
                    raise OSError
            self._stage("temp_write")
            descriptor = os.open(
                temporary.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                output.write(_canonical(value))
                output.flush()
                self._stage("temp_fsync")
                os.fsync(output.fileno())
            self._stage("atomic_replace")
            os.replace(
                temporary.name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            self._stage("parent_fsync")
            os.fsync(parent_descriptor)
        except Exception:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise StateWALError("WAL durable write failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _snapshot_hash_value(self, snapshot: AgentStateSnapshot) -> str:
        try:
            validated = AgentStateSnapshot.model_validate(
                snapshot.model_dump(mode="python")
            )
            payload = self._canonical_bytes(validated)
            return (
                self._snapshot_hash(validated)
                if self._snapshot_hash
                else hashlib.sha256(payload).hexdigest()
            )
        except Exception:
            raise StateWALFormatError("snapshot is invalid") from None

    def _generation_path(self, generation_id: UUID) -> Path:
        return self.root / "generations" / f"{generation_id}.jsonl"

    def _begin_generation(
        self,
        snapshot: AgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        reason: RecoveryReason = RecoveryReason.BOOTSTRAP,
        generation_id: UUID | None = None,
        predecessor_generation_id: UUID | None = None,
        predecessor_generation_hash: str | None = None,
        external_reconciliation_required: bool = False,
        prepared_recovery_id: UUID | None = None,
        allow_invalid_current: bool = False,
    ) -> Manifest:
        self._prepare_for_write()
        invalid_current_detected = False
        try:
            current = self.inspect_optional()
        except StateWALError:
            if prepared_recovery_id is None or not (
                external_reconciliation_required or allow_invalid_current
            ):
                raise
            current = None
            invalid_current_detected = True
        if (
            current is not None
            and current.active_manifest is not None
            and current.active_manifest.external_reconciliation_required
            and not external_reconciliation_required
        ):
            raise StateWALConflictError(
                "external reconciliation gate cannot be cleared by StateWAL"
            )
        generation = generation_id or uuid4()
        path = self._generation_path(generation)
        if path.exists():
            try:
                inspection = self.inspect()
            except StateWALError:
                inspection = None
            if (
                inspection is not None
                and inspection.active_manifest
                and inspection.active_manifest.active_generation_id == generation
                and inspection.latest_snapshot_sequence
                == snapshot.last_processed_event_sequence
                and inspection.latest_snapshot_hash
                == self._snapshot_hash_value(snapshot)
            ):
                return inspection.active_manifest
            try:
                baseline, baseline_hash = self._inspect_orphan_baseline(generation)
            except StateWALConflictError:
                if prepared_recovery_id is None:
                    raise
                self._preserve_partial_generation(path, prepared_recovery_id)
            else:
                if (
                    baseline.baseline_snapshot != snapshot
                    or baseline.journal_processing_high_water
                    != journal_processing_high_water
                    or baseline.reason is not reason
                    or baseline.predecessor_generation_id != predecessor_generation_id
                    or baseline.predecessor_generation_hash
                    != predecessor_generation_hash
                ):
                    raise StateWALConflictError("generation already exists")
                return self._publish_manifest(
                    baseline, baseline_hash, external_reconciliation_required
                )
        snapshot_hash = self._snapshot_hash_value(snapshot)
        baseline = BaselineRecord(
            record_type="baseline",
            schema_version=1,
            command_version=1,
            record_id=uuid4(),
            generation_id=generation,
            created_at=_now(),
            baseline_snapshot_sequence=snapshot.last_processed_event_sequence,
            baseline_snapshot_hash=snapshot_hash,
            baseline_snapshot=snapshot,
            journal_processing_high_water=journal_processing_high_water,
            predecessor_generation_id=predecessor_generation_id,
            predecessor_generation_hash=predecessor_generation_hash,
            reason=reason,
            previous_record_hash=None,
            record_hash="0" * 64,
        )
        raw, baseline_hash = _record_json(baseline, "kagya.state-wal.baseline")
        parent_descriptor = -1
        descriptor = -1
        try:
            self._stage("generation_write")
            parent_descriptor = self._open_private_dir(path.parent)
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                self._validate_descriptor(output.fileno())
                output.write(_canonical(raw))
                output.flush()
                self._stage("generation_fsync")
                os.fsync(output.fileno())
            os.fsync(parent_descriptor)
        except Exception:
            raise StateWALError("WAL generation write failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
        if invalid_current_detected:
            assert prepared_recovery_id is not None
            self._preserve_invalid_manifest(prepared_recovery_id)
        return self._publish_manifest(
            baseline, baseline_hash, external_reconciliation_required
        )

    def _preserve_partial_generation(self, path: Path, recovery_id: UUID) -> None:
        preserved = path.with_name(f".{path.name}.{recovery_id}.{uuid4()}.invalid")
        parent_descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(path.parent)
            os.replace(
                path.name,
                preserved.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except Exception:
            raise StateWALError("partial WAL generation cannot be preserved") from None
        finally:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _preserve_invalid_manifest(self, recovery_id: UUID) -> None:
        path = self.root / "manifest.json"
        preserved = self.root / f".manifest.json.{recovery_id}.{uuid4()}.invalid"
        parent_descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(self.root)
            os.replace(
                path.name,
                preserved.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except FileNotFoundError:
            return
        except Exception:
            raise StateWALError("invalid WAL manifest cannot be preserved") from None
        finally:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _publish_manifest(
        self,
        baseline: BaselineRecord,
        baseline_hash: str,
        external_reconciliation_required: bool,
    ) -> Manifest:
        manifest_base = Manifest(
            schema_version=1,
            command_version=1,
            active_generation_id=baseline.generation_id,
            active_baseline_record_id=baseline.record_id,
            active_baseline_record_hash=baseline_hash,
            predecessor_generation_id=baseline.predecessor_generation_id,
            predecessor_generation_hash=baseline.predecessor_generation_hash,
            external_reconciliation_required=external_reconciliation_required,
            manifest_hash="0" * 64,
        )
        manifest_raw = manifest_base.model_dump(mode="json")
        manifest_raw["manifest_hash"] = _domain_hash(
            "kagya.state-wal.manifest",
            {
                key: value
                for key, value in manifest_raw.items()
                if key != "manifest_hash"
            },
        )
        self._atomic_write(self.root / "manifest.json", manifest_raw)
        return Manifest.model_validate_json(_canonical(manifest_raw))

    def _inspect_orphan_baseline(
        self, generation_id: UUID
    ) -> tuple[BaselineRecord, str]:
        try:
            lines = self._read_regular(self._generation_path(generation_id)).splitlines(
                keepends=True
            )
            if len(lines) != 1:
                raise ValueError
            if not lines[0].endswith(b"\n"):
                raise ValueError
            raw = json.loads(lines[0])
            baseline = BaselineRecord.model_validate_json(_canonical(raw))
            record_hash = _record_json(baseline, "kagya.state-wal.baseline")[1]
            if (
                baseline.generation_id != generation_id
                or baseline.record_hash != record_hash
                or baseline.baseline_snapshot_sequence
                != baseline.baseline_snapshot.last_processed_event_sequence
                or baseline.baseline_snapshot_hash
                != self._snapshot_hash_value(baseline.baseline_snapshot)
            ):
                raise ValueError
            return baseline, record_hash
        except Exception:
            raise StateWALConflictError(
                "existing generation cannot be resumed"
            ) from None

    def begin_generation(
        self,
        snapshot: AgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        reason: RecoveryReason = RecoveryReason.BOOTSTRAP,
        generation_id: UUID | None = None,
        predecessor_generation_id: UUID | None = None,
        predecessor_generation_hash: str | None = None,
        external_reconciliation_required: bool = False,
    ) -> Manifest:
        with self._lock:
            return self._begin_generation(
                snapshot,
                journal_processing_high_water,
                reason=reason,
                generation_id=generation_id,
                predecessor_generation_id=predecessor_generation_id,
                predecessor_generation_hash=predecessor_generation_hash,
                external_reconciliation_required=external_reconciliation_required,
            )

    def resume_prepared_generation(
        self,
        snapshot: AgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        recovery_id: UUID,
        reason: RecoveryReason,
        generation_id: UUID,
        predecessor_generation_id: UUID | None,
        predecessor_generation_hash: str | None,
        external_reconciliation_required: bool,
    ) -> Manifest:
        """Resume only a Journal-prepared recovery while preserving partial bytes."""

        with self._lock:
            return self._begin_generation(
                snapshot,
                journal_processing_high_water,
                reason=reason,
                generation_id=generation_id,
                predecessor_generation_id=predecessor_generation_id,
                predecessor_generation_hash=predecessor_generation_hash,
                external_reconciliation_required=external_reconciliation_required,
                prepared_recovery_id=recovery_id,
            )

    def rebaseline_prepared_current(
        self,
        snapshot: AgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        recovery_id: UUID,
        generation_id: UUID,
        external_reconciliation_required: bool = False,
    ) -> Manifest:
        """Replace invalid active WAL metadata from Journal-confirmed current state."""

        with self._lock:
            return self._begin_generation(
                snapshot,
                journal_processing_high_water,
                reason=RecoveryReason.EXACT_CURRENT_REPAIR,
                generation_id=generation_id,
                predecessor_generation_id=None,
                predecessor_generation_hash=None,
                external_reconciliation_required=external_reconciliation_required,
                prepared_recovery_id=recovery_id,
                allow_invalid_current=True,
            )

    bootstrap = begin_generation

    def exists(self) -> bool:
        try:
            if not self._prepare_for_read():
                return False
            status = (self.root / "manifest.json").lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o600
            ):
                raise StateWALPermissionError("WAL manifest is unsafe")
            return True
        except FileNotFoundError:
            return False

    def inspect_optional(self) -> StateWALInspection:
        if not self._prepare_for_read():
            return StateWALInspection(
                exists=False,
                active_manifest=None,
                records=(),
                record_hashes=(),
                baseline_record_id=None,
                baseline_record_hash=None,
                latest_snapshot_sequence=None,
                latest_snapshot_hash=None,
            )
        try:
            manifest_raw = json.loads(self._read_regular(self.root / "manifest.json"))
        except StateWALMissing:
            return StateWALInspection(
                exists=False,
                active_manifest=None,
                records=(),
                record_hashes=(),
                baseline_record_id=None,
                baseline_record_hash=None,
                latest_snapshot_sequence=None,
                latest_snapshot_hash=None,
            )
        except StateWALError:
            raise
        except Exception:
            raise StateWALIntegrityError("WAL manifest is invalid") from None
        try:
            manifest = Manifest.model_validate_json(_canonical(manifest_raw))
            expected = _domain_hash(
                "kagya.state-wal.manifest",
                {
                    key: value
                    for key, value in manifest_raw.items()
                    if key != "manifest_hash"
                },
            )
            if manifest.manifest_hash != expected:
                raise ValueError
            records, hashes = self._read_generation(manifest)
        except StateWALMissing:
            raise StateWALIntegrityError("active WAL generation is absent") from None
        except StateWALError:
            raise
        except Exception:
            raise StateWALIntegrityError("WAL manifest is invalid") from None
        latest = records[-1]
        snapshot = (
            latest.baseline_snapshot
            if isinstance(latest, BaselineRecord)
            else latest.candidate_snapshot
        )
        return StateWALInspection(
            exists=True,
            active_manifest=manifest,
            records=tuple(records),
            record_hashes=tuple(hashes),
            baseline_record_id=records[0].record_id,
            baseline_record_hash=hashes[0],
            latest_snapshot_sequence=snapshot.last_processed_event_sequence,
            latest_snapshot_hash=self._snapshot_hash_value(snapshot),
        )

    def inspect(self) -> StateWALInspection:
        inspection = self.inspect_optional()
        if not inspection.exists:
            raise StateWALMissing("WAL is absent")
        return inspection

    def _read_generation(
        self, manifest: Manifest
    ) -> tuple[list[BaselineRecord | TransitionRecord], list[str]]:
        raw_lines = self._read_regular(
            self._generation_path(manifest.active_generation_id)
        ).splitlines(keepends=True)
        records: list[BaselineRecord | TransitionRecord] = []
        hashes: list[str] = []
        ids: set[UUID] = set()
        events: set[UUID] = set()
        state_sequence: int | None = None
        state_hash: str | None = None
        processing_sequence: int | None = None
        for line in raw_lines:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError
                raw = json.loads(line)
                kind = raw.get("record_type")
                model = (
                    BaselineRecord
                    if kind == "baseline"
                    else TransitionRecord
                    if kind == "transition"
                    else None
                )
                if model is None:
                    raise ValueError
                record = model.model_validate_json(_canonical(raw))
                record_hash = _record_json(record, f"kagya.state-wal.{kind}")[1]
                if (
                    not _HASH_RE.fullmatch(record.record_hash)
                    or record.record_hash != record_hash
                ):
                    raise ValueError
                if record.record_id in ids or (
                    isinstance(record, TransitionRecord) and record.event_id in events
                ):
                    raise ValueError
                if isinstance(record, TransitionRecord):
                    if (
                        state_sequence is None
                        or state_hash is None
                        or processing_sequence is None
                        or record.previous_record_hash != hashes[-1]
                        or record.prior_snapshot_sequence != state_sequence
                        or record.prior_snapshot_hash != state_hash
                        or record.processing_sequence <= processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.candidate_snapshot.last_processed_event_sequence
                        or record.candidate_snapshot_hash
                        != self._snapshot_hash_value(record.candidate_snapshot)
                    ):
                        raise ValueError
                    if record.candidate_snapshot_sequence <= state_sequence:
                        raise ValueError
                    events.add(record.event_id)
                else:
                    if records or record.previous_record_hash is not None:
                        raise ValueError
                    if (
                        record.baseline_snapshot_sequence
                        != record.baseline_snapshot.last_processed_event_sequence
                        or record.baseline_snapshot_hash
                        != self._snapshot_hash_value(record.baseline_snapshot)
                        or record.journal_processing_high_water
                        < record.baseline_snapshot_sequence
                    ):
                        raise ValueError
                if isinstance(record, TransitionRecord):
                    processing_sequence = record.processing_sequence
                    state_sequence = record.candidate_snapshot_sequence
                    state_hash = record.candidate_snapshot_hash
                else:
                    state_sequence = record.baseline_snapshot_sequence
                    state_hash = record.baseline_snapshot_hash
                    processing_sequence = record.journal_processing_high_water
                if record.generation_id != manifest.active_generation_id:
                    raise ValueError
                ids.add(record.record_id)
                records.append(record)
                hashes.append(record.record_hash)
            except Exception:
                raise StateWALIntegrityError("WAL generation is invalid") from None
        if (
            not records
            or records[0].record_id != manifest.active_baseline_record_id
            or hashes[0] != manifest.active_baseline_record_hash
        ):
            raise StateWALIntegrityError("WAL manifest does not anchor generation")
        return records, hashes

    def _append_transition(
        self,
        *,
        event_id: UUID,
        event_type: str,
        event_source: str,
        processing_sequence: int,
        prior_snapshot: AgentStateSnapshot,
        candidate_snapshot: AgentStateSnapshot,
    ) -> TransitionRecord:
        inspection = self.inspect()
        assert inspection.active_manifest is not None
        prior_hash = self._snapshot_hash_value(prior_snapshot)
        candidate_hash = self._snapshot_hash_value(candidate_snapshot)
        if (
            inspection.latest_snapshot_sequence
            != prior_snapshot.last_processed_event_sequence
            or inspection.latest_snapshot_hash != prior_hash
        ):
            raise StateWALConflictError("state lineage does not match WAL")
        if (
            candidate_snapshot.last_processed_event_sequence
            <= prior_snapshot.last_processed_event_sequence
        ):
            raise StateWALConflictError("state transition sequence must increase")
        if candidate_snapshot.last_processed_event_sequence != processing_sequence:
            raise StateWALConflictError(
                "candidate sequence must match processing sequence"
            )
        record = TransitionRecord(
            record_type="transition",
            schema_version=1,
            command_version=1,
            record_id=uuid4(),
            generation_id=inspection.active_manifest.active_generation_id,
            created_at=_now(),
            event_id=event_id,
            event_type=event_type,
            event_source=event_source,
            processing_sequence=processing_sequence,
            prior_snapshot_sequence=prior_snapshot.last_processed_event_sequence,
            prior_snapshot_hash=prior_hash,
            candidate_snapshot_sequence=candidate_snapshot.last_processed_event_sequence,
            candidate_snapshot_hash=candidate_hash,
            candidate_snapshot=candidate_snapshot,
            previous_record_hash=inspection.record_hashes[-1],
            record_hash="0" * 64,
        )
        raw, record_hash = _record_json(record, "kagya.state-wal.transition")
        record = record.model_copy(update={"record_hash": record_hash})
        path = self._generation_path(inspection.active_manifest.active_generation_id)
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(path.parent)
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "ab") as output:
                descriptor = -1
                self._validate_descriptor(output.fileno())
                output.write(_canonical(raw))
                output.flush()
                self._stage("transition_fsync")
                os.fsync(output.fileno())
            os.fsync(parent_descriptor)
        except Exception:
            raise StateWALError("WAL transition write failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
        return record

    def append_transition(self, **kwargs: Any) -> TransitionRecord:
        with self._lock:
            return self._append_transition(**kwargs)

    def publish_boot_anchor(
        self,
        *,
        snapshot_sequence: int,
        snapshot_hash: str,
        generation_id: UUID,
        anchored_record_id: UUID,
        anchored_record_hash: str,
        journal_processing_high_water: int,
        journal_tail_record_id: UUID | None = None,
        journal_tail_record_hash: str | None = None,
        journal_lineage_id: UUID,
    ) -> BootAnchor:
        inspection = self.inspect()
        if (
            inspection.active_manifest is None
            or generation_id != inspection.active_manifest.active_generation_id
        ):
            raise StateWALConflictError("boot anchor generation is not active")
        try:
            index = [record.record_id for record in inspection.records].index(
                anchored_record_id
            )
        except ValueError:
            raise StateWALConflictError("boot anchor record is not retained") from None
        record = inspection.records[index]
        snapshot = (
            record.baseline_snapshot
            if isinstance(record, BaselineRecord)
            else record.candidate_snapshot
        )
        if (
            snapshot.last_processed_event_sequence != snapshot_sequence
            or self._snapshot_hash_value(snapshot) != snapshot_hash
            or inspection.record_hashes[index] != anchored_record_hash
        ):
            raise StateWALIntegrityError("boot anchor does not match WAL")
        anchor_base = BootAnchor(
            schema_version=1,
            command_version=1,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            generation_id=generation_id,
            anchored_record_id=anchored_record_id,
            anchored_record_hash=anchored_record_hash,
            journal_processing_high_water=journal_processing_high_water,
            journal_tail_record_id=journal_tail_record_id,
            journal_tail_record_hash=journal_tail_record_hash,
            journal_lineage_id=journal_lineage_id,
            anchor_hash="0" * 64,
        )
        raw = anchor_base.model_dump(mode="json")
        raw["anchor_hash"] = _domain_hash(
            "kagya.state-wal.boot-anchor",
            {key: value for key, value in raw.items() if key != "anchor_hash"},
        )
        self._atomic_write(self.root / "boot_anchor.json", raw)
        return BootAnchor.model_validate_json(_canonical(raw))

    def inspect_boot_anchor(self) -> BootAnchor:
        try:
            raw = json.loads(self._read_regular(self.root / "boot_anchor.json"))
            anchor = BootAnchor.model_validate_json(_canonical(raw))
            expected = _domain_hash(
                "kagya.state-wal.boot-anchor",
                {key: value for key, value in raw.items() if key != "anchor_hash"},
            )
            if anchor.anchor_hash != expected:
                raise ValueError
            return anchor
        except Exception:
            raise StateWALIntegrityError("boot anchor is invalid") from None

    def inspect_boot_anchor_optional(self) -> BootAnchor | None:
        """Inspect an existing anchor without creating files."""

        if not self._prepare_for_read():
            return None
        try:
            (self.root / "boot_anchor.json").lstat()
        except FileNotFoundError:
            return None
        except OSError:
            raise StateWALIntegrityError("boot anchor cannot be inspected") from None
        return self.inspect_boot_anchor()

    def inspect_anchored_prefix(self) -> tuple[BootAnchor, AgentStateSnapshot]:
        anchor = self.inspect_boot_anchor()
        if not self._prepare_for_read():
            raise StateWALMissing("WAL is absent")
        path = self._generation_path(anchor.generation_id)
        lines = self._read_regular(path).splitlines(keepends=True)
        previous_hash: str | None = None
        previous_snapshot_sequence: int | None = None
        previous_state_hash: str | None = None
        previous_processing_sequence: int | None = None
        seen: set[UUID] = set()
        for line in lines:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError
                raw = json.loads(line)
                kind = raw.get("record_type")
                model = (
                    BaselineRecord
                    if kind == "baseline"
                    else TransitionRecord
                    if kind == "transition"
                    else None
                )
                if model is None:
                    raise ValueError
                record = model.model_validate_json(_canonical(raw))
                record_hash = _record_json(record, f"kagya.state-wal.{kind}")[1]
                if (
                    record.record_id in seen
                    or not _HASH_RE.fullmatch(record.record_hash)
                    or record.record_hash != record_hash
                ):
                    raise ValueError
                if record.generation_id != anchor.generation_id:
                    raise ValueError
                if not seen and not isinstance(record, BaselineRecord):
                    raise ValueError
                if isinstance(record, TransitionRecord):
                    if (
                        previous_snapshot_sequence is None
                        or previous_state_hash is None
                        or previous_processing_sequence is None
                        or record.previous_record_hash != previous_hash
                        or record.prior_snapshot_sequence != previous_snapshot_sequence
                        or record.prior_snapshot_hash != previous_state_hash
                        or record.processing_sequence <= previous_processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.candidate_snapshot.last_processed_event_sequence
                        or record.candidate_snapshot_hash
                        != self._snapshot_hash_value(record.candidate_snapshot)
                    ):
                        raise ValueError
                else:
                    if (
                        record.previous_record_hash is not None
                        or record.baseline_snapshot_sequence
                        != record.baseline_snapshot.last_processed_event_sequence
                        or record.baseline_snapshot_hash
                        != self._snapshot_hash_value(record.baseline_snapshot)
                        or record.journal_processing_high_water
                        < record.baseline_snapshot_sequence
                    ):
                        raise ValueError
                seen.add(record.record_id)
                snapshot = (
                    record.baseline_snapshot
                    if isinstance(record, BaselineRecord)
                    else record.candidate_snapshot
                )
                previous_snapshot_sequence = snapshot.last_processed_event_sequence
                previous_state_hash = (
                    record.candidate_snapshot_hash
                    if isinstance(record, TransitionRecord)
                    else record.baseline_snapshot_hash
                )
                previous_processing_sequence = (
                    record.processing_sequence
                    if isinstance(record, TransitionRecord)
                    else record.journal_processing_high_water
                )
                previous_hash = record.record_hash
                if record.record_id == anchor.anchored_record_id:
                    if (
                        record_hash != anchor.anchored_record_hash
                        or self._snapshot_hash_value(snapshot) != anchor.snapshot_hash
                    ):
                        raise ValueError
                    return anchor, snapshot
            except Exception:
                raise StateWALIntegrityError("anchored WAL prefix is invalid") from None
        raise StateWALConflictError("boot anchor record is not retained")

    def reconstruct(
        self,
        *,
        sequence: int | None = None,
        snapshot_hash: str | None = None,
        record_id: UUID | None = None,
    ) -> AgentStateSnapshot:
        inspection = self.inspect()
        found: AgentStateSnapshot | None = None
        for record in inspection.records:
            snapshot = (
                record.baseline_snapshot
                if isinstance(record, BaselineRecord)
                else record.candidate_snapshot
            )
            candidate_hash = (
                record.baseline_snapshot_hash
                if isinstance(record, BaselineRecord)
                else record.candidate_snapshot_hash
            )
            if (
                (sequence is None or snapshot.last_processed_event_sequence == sequence)
                and (snapshot_hash is None or candidate_hash == snapshot_hash)
                and (record_id is None or record.record_id == record_id)
            ):
                found = snapshot
        if found is None:
            raise StateWALConflictError("requested snapshot is not retained")
        return found

    def dry_run(
        self,
        *,
        sequence: int | None = None,
        snapshot_hash: str | None = None,
        record_id: UUID | None = None,
    ) -> DryRunDiff:
        inspection = self.inspect()
        current = self.reconstruct()
        target = self.reconstruct(
            sequence=sequence, snapshot_hash=snapshot_hash, record_id=record_id
        )
        before = current.model_dump(mode="json")
        after = target.model_dump(mode="json")
        changes = tuple(
            StateWALFieldChange(field=key, before=before.get(key), after=after.get(key))
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        )
        index = next(
            index
            for index, record in enumerate(inspection.records)
            if (
                record.baseline_snapshot
                if isinstance(record, BaselineRecord)
                else record.candidate_snapshot
            )
            == target
        )
        return DryRunDiff(
            target_snapshot_sequence=target.last_processed_event_sequence,
            target_snapshot_hash=self._snapshot_hash_value(target),
            current_snapshot_sequence=current.last_processed_event_sequence,
            current_snapshot_hash=self._snapshot_hash_value(current),
            changes=changes,
            records_replayed=index,
            external_side_effects_replayed=False,
        )

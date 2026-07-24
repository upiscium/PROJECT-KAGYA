"""Crash-safe behavioral artifact transactions, locking, and reconciliation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
import fcntl
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BehavioralArtifactStatus(StrEnum):
    VALID = "valid"
    PREPARED = "prepared"
    ORPHAN_RESULT = "orphan_result"
    ORPHAN_REGISTRY_REFERENCE = "orphan_registry_reference"
    HASH_MISMATCH = "hash_mismatch"
    CORRUPT = "corrupt"


class BehavioralEvaluationState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PREPARED = "prepared"
    FINALIZED = "finalized"
    RECONCILED = "reconciled"
    FAILED = "failed"


class BehavioralArtifactBusyError(RuntimeError):
    """Raised when another evaluation owns an adapter lease."""


class BehavioralArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: BehavioralArtifactStatus
    state: BehavioralEvaluationState = BehavioralEvaluationState.FINALIZED
    adapter_key: str | None = Field(default=None, max_length=128)
    failure_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def migrate_state(cls, value: Any) -> Any:
        if isinstance(value, dict) and "state" not in value:
            value = dict(value)
            value["state"] = (
                BehavioralEvaluationState.PREPARED
                if value.get("status") == BehavioralArtifactStatus.PREPARED.value
                else BehavioralEvaluationState.FINALIZED
            )
        return value


_THREAD_LOCKS_GUARD = Lock()
_THREAD_LOCKS: dict[str, Lock] = {}


def _thread_lock(path: Path) -> Lock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, Lock())


class BehavioralArtifactStore:
    """Keep privacy-safe relative references to immutable result artifacts."""

    def __init__(self, result_dir: Path) -> None:
        self.result_dir = result_dir.resolve()
        self.behavioral_dir = self.result_dir / "behavioral"
        self.registry_path = self.behavioral_dir / "artifact_registry.json"
        self.registry_lock_path = self.behavioral_dir / ".artifact_registry.lock"
        self.locks_dir = self.behavioral_dir / "locks"

    @contextmanager
    def adapter_lock(
        self, adapter_id: str, *, blocking: bool = False
    ) -> Iterator[None]:
        """Hold a cross-thread/process adapter lease for a complete evaluation saga."""

        key = hashlib.sha256(adapter_id.encode()).hexdigest()
        lock_path = self.locks_dir / f"{key}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        thread_lock = _thread_lock(lock_path)
        if not thread_lock.acquire(blocking=blocking):
            raise BehavioralArtifactBusyError("Behavioral evaluation already running")
        try:
            with lock_path.open("a+b") as lock_file:
                operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(lock_file.fileno(), operation)
                except BlockingIOError as exc:
                    raise BehavioralArtifactBusyError(
                        "Behavioral evaluation already running"
                    ) from exc
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            thread_lock.release()

    def begin(
        self, evaluation_id: str, *, adapter_key: str | None = None
    ) -> BehavioralArtifactRecord:
        """Reserve a globally unique evaluation ID before expensive generation."""

        self._validate_id(evaluation_id)
        with self._locked():
            records = self._registry_locked()
            if (
                evaluation_id in records
                or self.final_path(evaluation_id).exists()
                or self.prepared_path(evaluation_id).exists()
            ):
                raise ValueError(
                    f"Behavioral evaluation already exists: {evaluation_id}"
                )
            record = BehavioralArtifactRecord(
                evaluation_id=evaluation_id,
                relative_path=(Path("behavioral") / f"{evaluation_id}.json").as_posix(),
                sha256="0" * 64,
                status=BehavioralArtifactStatus.PREPARED,
                state=BehavioralEvaluationState.PENDING,
                adapter_key=adapter_key,
                updated_at=datetime.now(UTC),
            )
            records[evaluation_id] = record
            self._write_registry_locked(records)
            return record

    def mark_running(self, evaluation_id: str) -> BehavioralArtifactRecord:
        return self._transition(evaluation_id, BehavioralEvaluationState.RUNNING)

    def fail(self, evaluation_id: str, failure_code: str) -> BehavioralArtifactRecord:
        if (
            not failure_code
            or len(failure_code) > 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in failure_code
            )
        ):
            failure_code = "evaluation_failed"
        return self._transition(
            evaluation_id, BehavioralEvaluationState.FAILED, failure_code=failure_code
        )

    def commit(
        self, evaluation_id: str, payload: dict[str, Any]
    ) -> BehavioralArtifactRecord:
        self.prepare(evaluation_id, payload)
        return self.finalize(evaluation_id)

    def prepare(
        self, evaluation_id: str, payload: dict[str, Any]
    ) -> BehavioralArtifactRecord:
        """Atomically persist a prepared payload and its durable metadata."""

        self._validate_id(evaluation_id)
        content = json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(content).hexdigest()
        with self._locked():
            records = self._registry_locked()
            existing = records.get(evaluation_id)
            final_path = self.final_path(evaluation_id)
            prepared_path = self.prepared_path(evaluation_id)
            if (
                final_path.exists()
                or prepared_path.exists()
                or (
                    existing is not None
                    and existing.state
                    not in {
                        BehavioralEvaluationState.PENDING,
                        BehavioralEvaluationState.RUNNING,
                    }
                )
            ):
                raise ValueError(
                    f"Behavioral evaluation already exists: {evaluation_id}"
                )
            _atomic_write(prepared_path, content)
            record = BehavioralArtifactRecord(
                evaluation_id=evaluation_id,
                relative_path=(Path("behavioral") / f"{evaluation_id}.json").as_posix(),
                sha256=digest,
                status=BehavioralArtifactStatus.PREPARED,
                state=BehavioralEvaluationState.PREPARED,
                adapter_key=None if existing is None else existing.adapter_key,
                updated_at=datetime.now(UTC),
            )
            records[evaluation_id] = record
            self._write_registry_locked(records)
            return record

    def prepared_path(self, evaluation_id: str) -> Path:
        final_path = self.final_path(evaluation_id)
        return final_path.with_name(f".{final_path.name}.prepared")

    def final_path(self, evaluation_id: str) -> Path:
        return self.behavioral_dir / f"{evaluation_id}.json"

    def finalize(self, evaluation_id: str) -> BehavioralArtifactRecord:
        with self._locked():
            records = self._registry_locked()
            record = self._finalize_locked(evaluation_id, records)
            self._write_registry_locked(records)
            return record

    def _finalize_locked(
        self,
        evaluation_id: str,
        records: dict[str, BehavioralArtifactRecord],
    ) -> BehavioralArtifactRecord:
        record = records.get(evaluation_id)
        if record is None or record.state != BehavioralEvaluationState.PREPARED:
            raise ValueError("Behavioral artifact has no prepared transaction")
        final_path = self.result_dir / record.relative_path
        prepared_path = self.prepared_path(evaluation_id)
        content = prepared_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise ValueError("Prepared behavioral artifact hash mismatch")
        if not isinstance(json.loads(content), dict):
            raise ValueError("Prepared behavioral artifact is corrupt")
        os.replace(prepared_path, final_path)
        _fsync_directory(final_path.parent)
        record = record.model_copy(
            update={
                "status": BehavioralArtifactStatus.VALID,
                "state": BehavioralEvaluationState.FINALIZED,
                "updated_at": datetime.now(UTC),
            }
        )
        records[evaluation_id] = record
        return record

    def mark_reconciled(self, evaluation_id: str) -> BehavioralArtifactRecord:
        return self._transition(evaluation_id, BehavioralEvaluationState.RECONCILED)

    def reconcile(
        self, adapter_registry: Any | None = None, *, quarantine_invalid: bool = False
    ) -> tuple[BehavioralArtifactRecord, ...]:
        """Reconcile crash states while serializing every registry update."""

        with self._locked():
            return self._reconcile_locked(adapter_registry, quarantine_invalid)

    def _reconcile_locked(
        self, adapter_registry: Any | None, quarantine_invalid: bool
    ) -> tuple[BehavioralArtifactRecord, ...]:
        records = self._registry_locked()
        adapters = adapter_registry.list() if adapter_registry is not None else []
        bindings: dict[str, Any] = {}
        for entry in adapters:
            if entry.behavioral_evaluation_id is not None:
                bindings[entry.behavioral_evaluation_id] = entry
            if entry.real_model_behavioral_evaluation_id is not None:
                bindings[entry.real_model_behavioral_evaluation_id] = SimpleNamespace(
                    adapter_id=entry.adapter_id,
                    adapter_hash=entry.adapter_hash,
                    base_model_revision=entry.base_model_revision,
                    behavioral_evaluation_id=entry.real_model_behavioral_evaluation_id,
                    behavioral_evaluation_path=entry.real_model_behavioral_evaluation_path,
                    behavioral_result_hash=entry.real_model_behavioral_result_hash,
                    behavioral_artifact_state=entry.real_model_behavioral_artifact_state,
                )
        known_paths = {record.relative_path for record in records.values()}
        reconciled: dict[str, BehavioralArtifactRecord] = {}
        for evaluation_id, original in records.items():
            record = original
            path = self.result_dir / record.relative_path
            prepared = self.prepared_path(evaluation_id)
            binding = bindings.get(evaluation_id)
            # A running state has no durable ownership after restart/reconcile.
            if record.state in {
                BehavioralEvaluationState.PENDING,
                BehavioralEvaluationState.RUNNING,
            } and not self._adapter_lease_active(record.adapter_key):
                record = record.model_copy(
                    update={
                        "state": BehavioralEvaluationState.FAILED,
                        "failure_code": "interrupted_before_prepare",
                        "status": BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE,
                        "updated_at": datetime.now(UTC),
                    }
                )
            if (
                record.state == BehavioralEvaluationState.PREPARED
                and prepared.is_file()
                and binding is not None
                and binding.behavioral_artifact_state == "prepared"
            ):
                try:
                    record = self._finalize_locked(evaluation_id, records)
                    path = self.result_dir / record.relative_path
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    pass
            status = record.status
            if record.state == BehavioralEvaluationState.PREPARED:
                status = BehavioralArtifactStatus.PREPARED
                if not prepared.is_file():
                    status = BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE
                else:
                    status = self._file_status(prepared, record.sha256, prepared=True)
            elif path.is_file():
                status = self._file_status(path, record.sha256)
            elif record.state != BehavioralEvaluationState.FAILED:
                status = BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE
            if adapter_registry is not None:
                if binding is None and status == BehavioralArtifactStatus.VALID:
                    status = BehavioralArtifactStatus.ORPHAN_RESULT
                elif binding is not None and status not in {
                    BehavioralArtifactStatus.CORRUPT,
                    BehavioralArtifactStatus.PREPARED,
                    BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE,
                }:
                    status = self._cross_registry_status(record, path, binding)
                if (
                    binding is not None
                    and quarantine_invalid
                    and status
                    in {
                        BehavioralArtifactStatus.HASH_MISMATCH,
                        BehavioralArtifactStatus.CORRUPT,
                        BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE,
                    }
                    and binding.behavioral_artifact_state != "quarantined"
                ):
                    adapter_registry.quarantine_behavioral_evaluation(
                        binding.adapter_id, evaluation_id=evaluation_id
                    )
            state = record.state
            if (
                status == BehavioralArtifactStatus.VALID
                and binding is not None
                and binding.behavioral_artifact_state == "reconciled"
            ):
                state = BehavioralEvaluationState.RECONCILED
            reconciled[evaluation_id] = record.model_copy(
                update={
                    "status": status,
                    "state": state,
                    "updated_at": record.updated_at
                    if status == original.status and state == original.state
                    else datetime.now(UTC),
                }
            )
        if self.behavioral_dir.exists():
            for path in self.behavioral_dir.glob("*.json"):
                relative = path.relative_to(self.result_dir).as_posix()
                if path == self.registry_path or relative in known_paths:
                    continue
                identifier = path.stem
                try:
                    content = path.read_bytes()
                    if not isinstance(json.loads(content), dict):
                        raise ValueError("artifact root is not an object")
                    digest = hashlib.sha256(content).hexdigest()
                    status = (
                        BehavioralArtifactStatus.HASH_MISMATCH
                        if identifier in bindings
                        else BehavioralArtifactStatus.ORPHAN_RESULT
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    digest = "0" * 64
                    status = BehavioralArtifactStatus.CORRUPT
                reconciled[identifier] = BehavioralArtifactRecord(
                    evaluation_id=identifier,
                    relative_path=relative,
                    sha256=digest,
                    status=status,
                    state=BehavioralEvaluationState.FAILED,
                    failure_code="orphan_artifact",
                    updated_at=datetime.now(UTC),
                )
        for evaluation_id, binding in bindings.items():
            if evaluation_id in reconciled:
                continue
            if (
                quarantine_invalid
                and adapter_registry is not None
                and binding.behavioral_artifact_state != "quarantined"
            ):
                adapter_registry.quarantine_behavioral_evaluation(
                    binding.adapter_id, evaluation_id=evaluation_id
                )
            reconciled[evaluation_id] = BehavioralArtifactRecord(
                evaluation_id=evaluation_id,
                relative_path=(Path("behavioral") / f"{evaluation_id}.json").as_posix(),
                sha256=binding.behavioral_result_hash or "0" * 64,
                status=BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE,
                state=BehavioralEvaluationState.FAILED,
                failure_code="missing_artifact",
                updated_at=datetime.now(UTC),
            )
        self._write_registry_locked(reconciled)
        return tuple(sorted(reconciled.values(), key=lambda item: item.evaluation_id))

    def _adapter_lease_active(self, adapter_key: str | None) -> bool:
        if adapter_key is None:
            return False
        try:
            with self.adapter_lock(adapter_key, blocking=False):
                return False
        except BehavioralArtifactBusyError:
            return True

    def _file_status(
        self, path: Path, digest: str, *, prepared: bool = False
    ) -> BehavioralArtifactStatus:
        try:
            content = path.read_bytes()
            if not isinstance(json.loads(content), dict):
                raise ValueError("artifact root is not an object")
            if hashlib.sha256(content).hexdigest() != digest:
                return BehavioralArtifactStatus.HASH_MISMATCH
            return (
                BehavioralArtifactStatus.PREPARED
                if prepared
                else BehavioralArtifactStatus.VALID
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return BehavioralArtifactStatus.CORRUPT

    def _cross_registry_status(
        self, record: BehavioralArtifactRecord, path: Path, binding: Any
    ) -> BehavioralArtifactStatus:
        if binding.behavioral_artifact_state == "prepared":
            return BehavioralArtifactStatus.PREPARED
        if binding.behavioral_artifact_state not in {"finalized", "reconciled"}:
            return BehavioralArtifactStatus.HASH_MISMATCH
        try:
            content = path.read_bytes()
            payload = json.loads(content)
            manifest = payload.get("manifest") if isinstance(payload, dict) else None
            if (
                not isinstance(manifest, dict)
                or payload.get("evaluation_id") != record.evaluation_id
                or manifest.get("candidate_adapter_id") != binding.adapter_id
                or manifest.get("candidate_adapter_hash") != binding.adapter_hash
                or (
                    manifest.get("base_model_revision_resolved")
                    or manifest.get("base_model_revision")
                )
                != binding.base_model_revision
                or path.resolve()
                != Path(binding.behavioral_evaluation_path or "").resolve()
                or hashlib.sha256(content).hexdigest() != record.sha256
                or record.sha256 != binding.behavioral_result_hash
            ):
                return BehavioralArtifactStatus.HASH_MISMATCH
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return BehavioralArtifactStatus.CORRUPT
        return BehavioralArtifactStatus.VALID

    def valid(self, evaluation_id: str) -> BehavioralArtifactRecord | None:
        return next(
            (
                item
                for item in self.reconcile()
                if item.evaluation_id == evaluation_id
                and item.status == BehavioralArtifactStatus.VALID
            ),
            None,
        )

    def _transition(
        self,
        evaluation_id: str,
        state: BehavioralEvaluationState,
        *,
        failure_code: str | None = None,
    ) -> BehavioralArtifactRecord:
        with self._locked():
            records = self._registry_locked()
            record = records.get(evaluation_id)
            if record is None:
                raise ValueError(f"Unknown behavioral evaluation: {evaluation_id}")
            record = record.model_copy(
                update={
                    "state": state,
                    "failure_code": failure_code,
                    "updated_at": datetime.now(UTC),
                }
            )
            records[evaluation_id] = record
            self._write_registry_locked(records)
            return record

    def _validate_id(self, evaluation_id: str) -> None:
        if not evaluation_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in evaluation_id
        ):
            raise ValueError("Unsafe behavioral evaluation ID")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.behavioral_dir.mkdir(parents=True, exist_ok=True)
        thread_lock = _thread_lock(self.registry_lock_path)
        with thread_lock, self.registry_lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _registry_locked(self) -> dict[str, BehavioralArtifactRecord]:
        if not self.registry_path.exists():
            return {}
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            values = raw.get("artifacts", []) if isinstance(raw, dict) else []
            return {
                item.evaluation_id: item
                for item in (
                    BehavioralArtifactRecord.model_validate(value) for value in values
                )
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Behavioral artifact registry is corrupt") from exc

    def _write_registry_locked(
        self, records: dict[str, BehavioralArtifactRecord]
    ) -> None:
        payload = {
            "schema_version": 10,
            "artifacts": [
                item.model_dump(mode="json")
                for item in sorted(
                    records.values(), key=lambda value: value.evaluation_id
                )
            ],
        }
        _atomic_write(
            self.registry_path,
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode(),
        )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

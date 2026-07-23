"""Crash-safe behavioral artifact transaction and reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BehavioralArtifactStatus(StrEnum):
    VALID = "valid"
    PREPARED = "prepared"
    ORPHAN_RESULT = "orphan_result"
    ORPHAN_REGISTRY_REFERENCE = "orphan_registry_reference"
    HASH_MISMATCH = "hash_mismatch"
    CORRUPT = "corrupt"


class BehavioralArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: BehavioralArtifactStatus
    updated_at: datetime


class BehavioralArtifactStore:
    """Keep privacy-safe relative references to immutable result artifacts."""

    def __init__(self, result_dir: Path) -> None:
        self.result_dir = result_dir.resolve()
        self.behavioral_dir = self.result_dir / "behavioral"
        self.registry_path = self.behavioral_dir / "artifact_registry.json"

    def commit(
        self, evaluation_id: str, payload: dict[str, Any]
    ) -> BehavioralArtifactRecord:
        self.prepare(evaluation_id, payload)
        return self.finalize(evaluation_id)

    def prepare(
        self, evaluation_id: str, payload: dict[str, Any]
    ) -> BehavioralArtifactRecord:
        """Atomically persist a prepared payload and its durable metadata."""

        if not evaluation_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in evaluation_id
        ):
            raise ValueError("Unsafe behavioral evaluation ID")
        relative = Path("behavioral") / f"{evaluation_id}.json"
        final_path = self.result_dir / relative
        prepared_path = final_path.with_name(f".{final_path.name}.prepared")
        records = self._registry()
        if evaluation_id in records or final_path.exists() or prepared_path.exists():
            raise ValueError(f"Behavioral evaluation already exists: {evaluation_id}")
        content = json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(content).hexdigest()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(prepared_path, content)
        record = BehavioralArtifactRecord(
            evaluation_id=evaluation_id,
            relative_path=relative.as_posix(),
            sha256=digest,
            status=BehavioralArtifactStatus.PREPARED,
            updated_at=datetime.now(UTC),
        )
        records[evaluation_id] = record
        self._write_registry(records)
        return record

    def prepared_path(self, evaluation_id: str) -> Path:
        final_path = self.result_dir / "behavioral" / f"{evaluation_id}.json"
        return final_path.with_name(f".{final_path.name}.prepared")

    def final_path(self, evaluation_id: str) -> Path:
        return self.result_dir / "behavioral" / f"{evaluation_id}.json"

    def finalize(self, evaluation_id: str) -> BehavioralArtifactRecord:
        """Promote a complete prepared payload to its immutable final path."""

        records = self._registry()
        record = records.get(evaluation_id)
        if record is None or record.status != BehavioralArtifactStatus.PREPARED:
            raise ValueError("Behavioral artifact has no prepared transaction")
        final_path = self.result_dir / record.relative_path
        prepared_path = self.prepared_path(evaluation_id)
        content = prepared_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise ValueError("Prepared behavioral artifact hash mismatch")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Prepared behavioral artifact is corrupt")
        os.replace(prepared_path, final_path)
        _fsync_directory(final_path.parent)
        record = record.model_copy(
            update={
                "status": BehavioralArtifactStatus.VALID,
                "updated_at": datetime.now(UTC),
            }
        )
        records[evaluation_id] = record
        self._write_registry(records)
        return record

    def reconcile(
        self, adapter_registry: Any | None = None, *, quarantine_invalid: bool = False
    ) -> tuple[BehavioralArtifactRecord, ...]:
        """Reconcile files, artifact metadata, and AdapterRegistry bindings."""

        records = self._registry()
        adapters = adapter_registry.list() if adapter_registry is not None else []
        bindings = {
            entry.behavioral_evaluation_id: entry
            for entry in adapters
            if entry.behavioral_evaluation_id is not None
        }
        known_paths = {record.relative_path for record in records.values()}
        reconciled: dict[str, BehavioralArtifactRecord] = {}
        for evaluation_id, record in records.items():
            path = self.result_dir / record.relative_path
            binding = bindings.get(evaluation_id)
            prepared = path.with_name(f".{path.name}.prepared")
            if (
                record.status == BehavioralArtifactStatus.PREPARED
                and prepared.is_file()
                and binding is not None
                and binding.behavioral_artifact_state == "prepared"
            ):
                try:
                    record = self.finalize(evaluation_id)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    pass
            status = BehavioralArtifactStatus.VALID
            if record.status == BehavioralArtifactStatus.PREPARED:
                if prepared.is_file() and not path.exists():
                    status = BehavioralArtifactStatus.PREPARED
                    try:
                        prepared_content = prepared.read_bytes()
                        prepared_value = json.loads(prepared_content)
                        if not isinstance(prepared_value, dict):
                            raise ValueError("prepared artifact root is not an object")
                        if (
                            hashlib.sha256(prepared_content).hexdigest()
                            != record.sha256
                        ):
                            status = BehavioralArtifactStatus.HASH_MISMATCH
                    except (
                        OSError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        ValueError,
                    ):
                        status = BehavioralArtifactStatus.CORRUPT
                elif not path.is_file():
                    status = BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE
            elif not path.is_file():
                status = BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE
            if path.is_file():
                try:
                    content = path.read_bytes()
                    value = json.loads(content)
                    if not isinstance(value, dict):
                        raise ValueError("artifact root is not an object")
                    if hashlib.sha256(content).hexdigest() != record.sha256:
                        status = BehavioralArtifactStatus.HASH_MISMATCH
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    status = BehavioralArtifactStatus.CORRUPT
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
                        binding.adapter_id
                    )
            reconciled[evaluation_id] = record.model_copy(
                update={
                    "status": status,
                    "updated_at": (
                        record.updated_at
                        if status == record.status
                        else datetime.now(UTC)
                    ),
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
                    value = json.loads(content)
                    if not isinstance(value, dict):
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
                adapter_registry.quarantine_behavioral_evaluation(binding.adapter_id)
            missing_relative = Path("behavioral") / f"{evaluation_id}.json"
            reconciled[evaluation_id] = BehavioralArtifactRecord(
                evaluation_id=evaluation_id,
                relative_path=missing_relative.as_posix(),
                sha256=binding.behavioral_result_hash or "0" * 64,
                status=BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE,
                updated_at=datetime.now(UTC),
            )
        self._write_registry(reconciled)
        return tuple(sorted(reconciled.values(), key=lambda item: item.evaluation_id))

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
            expected_path = path.resolve()
            bound_path = Path(binding.behavioral_evaluation_path or "").resolve()
            if (
                not isinstance(manifest, dict)
                or payload.get("evaluation_id") != record.evaluation_id
                or manifest.get("candidate_adapter_id") != binding.adapter_id
                or manifest.get("candidate_adapter_hash") != binding.adapter_hash
                or manifest.get("base_model_revision") != binding.base_model_revision
                or expected_path != bound_path
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

    def _registry(self) -> dict[str, BehavioralArtifactRecord]:
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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {}

    def _write_registry(self, records: dict[str, BehavioralArtifactRecord]) -> None:
        payload = {
            "schema_version": 1,
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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

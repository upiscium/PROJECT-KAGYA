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

    def reconcile(self) -> tuple[BehavioralArtifactRecord, ...]:
        records = self._registry()
        known_paths = {record.relative_path for record in records.values()}
        reconciled: dict[str, BehavioralArtifactRecord] = {}
        for evaluation_id, record in records.items():
            path = self.result_dir / record.relative_path
            status = BehavioralArtifactStatus.VALID
            if record.status == BehavioralArtifactStatus.PREPARED:
                prepared = path.with_name(f".{path.name}.prepared")
                if prepared.is_file() and not path.exists():
                    status = BehavioralArtifactStatus.PREPARED
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
            reconciled[evaluation_id] = record.model_copy(
                update={"status": status, "updated_at": datetime.now(UTC)}
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
                    status = BehavioralArtifactStatus.ORPHAN_RESULT
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
        self._write_registry(reconciled)
        return tuple(sorted(reconciled.values(), key=lambda item: item.evaluation_id))

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

"""JSON-backed adapter lifecycle registry."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from kagya.config import Settings


class AdapterStatus(StrEnum):
    CANDIDATE = "candidate"
    TRIAL_ACTIVE = "trial_active"
    APPROVED = "approved"
    ACTIVE = "active"
    REJECTED = "rejected"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class AdapterEntry:
    adapter_id: str
    base_model: str
    path: str
    status: AdapterStatus
    dataset_path: str
    dataset_hash: str
    eval_score: float | None = None
    eval_result_path: str | None = None
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["state"] = self.status.value
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AdapterEntry":
        status = AdapterStatus(str(data.get("status", data.get("state"))))
        return cls(
            adapter_id=str(data["adapter_id"]),
            base_model=str(data["base_model"]),
            path=str(data["path"]),
            status=status,
            dataset_path=str(data["dataset_path"]),
            dataset_hash=str(data["dataset_hash"]),
            eval_score=_optional_float(data.get("eval_score")),
            eval_result_path=_optional_str(data.get("eval_result_path")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            notes=str(data.get("notes", "")),
        )


class AdapterRegistry:
    """Persist and validate adapter lifecycle transitions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.adapter_registry.path
        self._lock_path = Path(f"{self.path}.lock")

    def register_candidate(
        self,
        *,
        adapter_id: str,
        adapter_path: str | Path,
        dataset_path: str | Path,
        dataset_hash: str,
        base_model: str | None = None,
        notes: str = "",
    ) -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            if self._lookup_locked(entries, adapter_id) is not None:
                raise ValueError(f"Adapter already registered: {adapter_id}")
            now = _now_iso()
            entry = AdapterEntry(
                adapter_id=adapter_id,
                base_model=base_model or self.settings.model.primary_id,
                path=str(adapter_path),
                status=AdapterStatus.CANDIDATE,
                dataset_path=str(dataset_path),
                dataset_hash=dataset_hash,
                created_at=now,
                updated_at=now,
                notes=notes,
            )
            entries.append(entry)
            self._write_locked(entries)
            return entry

    def list(self) -> list[AdapterEntry]:
        with self._locked(exclusive=False):
            return self._list_locked()

    def _list_locked(self) -> list[AdapterEntry]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as registry_file:
            data = json.load(registry_file)
        adapters = data.get("adapters", []) if isinstance(data, dict) else []
        return [AdapterEntry.from_json(item) for item in adapters if isinstance(item, dict)]

    def lookup(self, adapter_id: str) -> AdapterEntry | None:
        with self._locked(exclusive=False):
            return self._lookup_locked(self._list_locked(), adapter_id)

    def apply_evaluation(
        self,
        adapter_id: str,
        *,
        score: float,
        result_path: str | Path,
        next_status: AdapterStatus | None = None,
    ) -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            if entry.status != AdapterStatus.CANDIDATE:
                raise ValueError("Only candidate adapters can receive evaluation gating")
            if next_status is None:
                if score >= self.settings.adapter_registry.trial_threshold:
                    next_status = AdapterStatus.TRIAL_ACTIVE
                elif score < self.settings.adapter_registry.reject_threshold:
                    next_status = AdapterStatus.REJECTED
                else:
                    next_status = AdapterStatus.CANDIDATE
            if next_status not in {
                AdapterStatus.CANDIDATE,
                AdapterStatus.TRIAL_ACTIVE,
                AdapterStatus.REJECTED,
            }:
                raise ValueError("Invalid evaluation status")
            return self._replace_locked(
                entries,
                adapter_id,
                status=next_status,
                eval_score=score,
                eval_result_path=str(result_path),
            )

    def approve(self, adapter_id: str, *, notes: str = "") -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            self._ensure_transition(entry.status, AdapterStatus.APPROVED)
            return self._replace_locked(
                entries,
                adapter_id,
                status=AdapterStatus.APPROVED,
                notes=notes or entry.notes,
            )

    def activate(self, adapter_id: str) -> AdapterEntry:
        with self._locked(exclusive=True):
            current_entries = self._list_locked()
            entry = self._require_locked(current_entries, adapter_id)
            self._ensure_transition(entry.status, AdapterStatus.ACTIVE)
            entries = []
            activated: AdapterEntry | None = None
            now = _now_iso()
            for existing in current_entries:
                if existing.adapter_id == adapter_id:
                    activated = _copy_entry(existing, status=AdapterStatus.ACTIVE, updated_at=now)
                    entries.append(activated)
                elif existing.status == AdapterStatus.ACTIVE:
                    entries.append(_copy_entry(existing, status=AdapterStatus.ARCHIVED, updated_at=now))
                else:
                    entries.append(existing)
            assert activated is not None
            self._write_locked(entries)
            return activated

    def transition(self, adapter_id: str, status: AdapterStatus) -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            self._ensure_transition(entry.status, status)
            return self._replace_locked(entries, adapter_id, status=status)

    def _lookup_locked(
        self, entries: list[AdapterEntry], adapter_id: str
    ) -> AdapterEntry | None:
        return next((entry for entry in entries if entry.adapter_id == adapter_id), None)

    def _require_locked(self, entries: list[AdapterEntry], adapter_id: str) -> AdapterEntry:
        entry = self._lookup_locked(entries, adapter_id)
        if entry is None:
            raise ValueError(f"Unknown adapter: {adapter_id}")
        return entry

    def _replace_locked(
        self, current_entries: list[AdapterEntry], adapter_id: str, **updates: Any
    ) -> AdapterEntry:
        entries: list[AdapterEntry] = []
        updated_entry: AdapterEntry | None = None
        for entry in current_entries:
            if entry.adapter_id != adapter_id:
                entries.append(entry)
                continue
            updated_entry = _copy_entry(entry, updated_at=_now_iso(), **updates)
            entries.append(updated_entry)
        if updated_entry is None:
            raise ValueError(f"Unknown adapter: {adapter_id}")
        self._write_locked(entries)
        return updated_entry

    def _write_locked(self, entries: list[AdapterEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as registry_file:
                os.fchmod(registry_file.fileno(), 0o600)
                json.dump(
                    {"adapters": [entry.to_json() for entry in entries]},
                    registry_file,
                    indent=2,
                )
                registry_file.flush()
                os.fsync(registry_file.fileno())
            os.replace(temp_path, self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temp_path.unlink(missing_ok=True)

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock_file:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _ensure_transition(self, current: AdapterStatus, target: AdapterStatus) -> None:
        allowed = {
            AdapterStatus.CANDIDATE: {AdapterStatus.TRIAL_ACTIVE, AdapterStatus.REJECTED},
            AdapterStatus.TRIAL_ACTIVE: {AdapterStatus.APPROVED},
            AdapterStatus.APPROVED: {AdapterStatus.ACTIVE},
            AdapterStatus.ACTIVE: {AdapterStatus.ARCHIVED},
            AdapterStatus.REJECTED: set(),
            AdapterStatus.ARCHIVED: set(),
        }
        if target not in allowed[current]:
            raise ValueError(f"Invalid adapter status transition: {current} -> {target}")


def _copy_entry(entry: AdapterEntry, **updates: Any) -> AdapterEntry:
    data = asdict(entry)
    data.update(updates)
    return AdapterEntry(**data)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)

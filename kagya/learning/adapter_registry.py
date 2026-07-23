"""JSON-backed adapter lifecycle registry."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import builtins
import fcntl
import hashlib
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
    base_model_revision: str | None = None
    adapter_hash: str | None = None
    parent_adapter_id: str | None = None
    parent_adapter_hash: str | None = None
    training_job_id: str | None = None
    training_node_id: str | None = None
    submitted_by_node_id: str | None = None
    imported_by_node_id: str | None = None
    training_manifest_path: str | None = None
    worker_evaluation_path: str | None = None
    local_evaluation_path: str | None = None
    activation_sequence: int | None = None
    evaluation_set_hashes: tuple[str, ...] = ()
    evaluation_dataset_path: str | None = None
    dataset_record_hashes: tuple[str, ...] = ()
    dataset_repetition_count: int = 0
    dataset_overlap_count: int = 0
    dataset_overlap_ratio: float = 0.0
    holdout_score: float | None = None
    holdout_baseline_score: float | None = None
    holdout_regression: bool = False
    drift_scores: dict[str, float] | None = None
    activation_gate_passed: bool = False
    behavioral_evaluation_id: str | None = None
    behavioral_evaluation_path: str | None = None
    behavioral_gate_passed: bool | None = None
    rollout_state: str = "candidate"
    canary_failures: int = 0
    rollback_target_id: str | None = None
    schema_version: int = 3

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
            base_model_revision=_optional_str(data.get("base_model_revision")),
            adapter_hash=_optional_str(data.get("adapter_hash")),
            parent_adapter_id=_optional_str(data.get("parent_adapter_id")),
            parent_adapter_hash=_optional_str(data.get("parent_adapter_hash")),
            training_job_id=_optional_str(data.get("training_job_id")),
            training_node_id=_optional_str(data.get("training_node_id")),
            submitted_by_node_id=_optional_str(data.get("submitted_by_node_id")),
            imported_by_node_id=_optional_str(data.get("imported_by_node_id")),
            training_manifest_path=_optional_str(data.get("training_manifest_path")),
            worker_evaluation_path=_optional_str(data.get("worker_evaluation_path")),
            local_evaluation_path=_optional_str(data.get("local_evaluation_path")),
            activation_sequence=(
                None
                if data.get("activation_sequence") is None
                else int(data["activation_sequence"])
            ),
            evaluation_set_hashes=tuple(
                str(item) for item in data.get("evaluation_set_hashes", ())
            ),
            evaluation_dataset_path=_optional_str(data.get("evaluation_dataset_path")),
            dataset_record_hashes=tuple(
                str(item) for item in data.get("dataset_record_hashes", ())
            ),
            dataset_repetition_count=int(data.get("dataset_repetition_count", 0)),
            dataset_overlap_count=int(data.get("dataset_overlap_count", 0)),
            dataset_overlap_ratio=float(data.get("dataset_overlap_ratio", 0.0)),
            holdout_score=_optional_float(data.get("holdout_score")),
            holdout_baseline_score=_optional_float(data.get("holdout_baseline_score")),
            holdout_regression=bool(data.get("holdout_regression", False)),
            drift_scores={
                str(key): float(value)
                for key, value in data.get("drift_scores", {}).items()
            }
            if isinstance(data.get("drift_scores"), dict)
            else None,
            activation_gate_passed=bool(data.get("activation_gate_passed", False)),
            behavioral_evaluation_id=_optional_str(data.get("behavioral_evaluation_id")),
            behavioral_evaluation_path=_optional_str(data.get("behavioral_evaluation_path")),
            behavioral_gate_passed=(
                None
                if data.get("behavioral_gate_passed") is None
                else bool(data["behavioral_gate_passed"])
            ),
            rollout_state=str(data.get("rollout_state", "candidate")),
            canary_failures=int(data.get("canary_failures", 0)),
            rollback_target_id=_optional_str(data.get("rollback_target_id")),
            schema_version=int(data.get("schema_version", 1)),
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
        base_model_revision: str | None = None,
        adapter_hash: str | None = None,
        parent_adapter_id: str | None = None,
        parent_adapter_hash: str | None = None,
        training_job_id: str | None = None,
        training_node_id: str | None = None,
        submitted_by_node_id: str | None = None,
        imported_by_node_id: str | None = None,
        training_manifest_path: str | None = None,
        worker_evaluation_path: str | None = None,
        local_evaluation_path: str | None = None,
        evaluation_set_hashes: tuple[str, ...] = (),
        evaluation_dataset_path: str | Path | None = None,
    ) -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            if self._lookup_locked(entries, adapter_id) is not None:
                raise ValueError(f"Adapter already registered: {adapter_id}")
            if adapter_hash is not None and any(
                entry.adapter_hash == adapter_hash for entry in entries
            ):
                raise ValueError(f"Adapter hash already registered: {adapter_hash}")
            if training_job_id is not None and any(
                entry.training_job_id == training_job_id for entry in entries
            ):
                raise ValueError(f"Training job already registered: {training_job_id}")
            self._validate_lineage_locked(
                entries,
                adapter_id=adapter_id,
                base_model=base_model or self.settings.model.primary_id,
                base_model_revision=base_model_revision,
                parent_adapter_id=parent_adapter_id,
                parent_adapter_hash=parent_adapter_hash,
            )
            record_hashes, repetitions = _dataset_record_hashes(Path(dataset_path))
            ancestor_hashes: set[str] = set()
            if parent_adapter_id is not None:
                for ancestor in self._lineage_locked(entries, parent_adapter_id):
                    ancestor_hashes.update(ancestor.dataset_record_hashes)
            overlap = len(set(record_hashes) & ancestor_hashes)
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
                base_model_revision=base_model_revision,
                adapter_hash=adapter_hash,
                parent_adapter_id=parent_adapter_id,
                parent_adapter_hash=parent_adapter_hash,
                training_job_id=training_job_id,
                training_node_id=training_node_id,
                submitted_by_node_id=submitted_by_node_id,
                imported_by_node_id=imported_by_node_id,
                training_manifest_path=training_manifest_path,
                worker_evaluation_path=worker_evaluation_path,
                local_evaluation_path=local_evaluation_path,
                evaluation_set_hashes=evaluation_set_hashes,
                evaluation_dataset_path=(
                    None
                    if evaluation_dataset_path is None
                    else str(evaluation_dataset_path)
                ),
                dataset_record_hashes=record_hashes,
                dataset_repetition_count=repetitions,
                dataset_overlap_count=overlap,
                dataset_overlap_ratio=overlap / len(set(record_hashes))
                if record_hashes
                else 0.0,
            )
            entries.append(entry)
            self._write_locked(entries)
            return entry

    def list(self) -> list[AdapterEntry]:
        with self._locked(exclusive=False):
            return self._list_locked()

    def _list_locked(self) -> builtins.list[AdapterEntry]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as registry_file:
            data = json.load(registry_file)
        adapters = data.get("adapters", []) if isinstance(data, dict) else []
        return [
            AdapterEntry.from_json(item) for item in adapters if isinstance(item, dict)
        ]

    def lookup(self, adapter_id: str) -> AdapterEntry | None:
        with self._locked(exclusive=False):
            return self._lookup_locked(self._list_locked(), adapter_id)

    def lineage(self, adapter_id: str) -> builtins.list[AdapterEntry]:
        with self._locked(exclusive=False):
            entries = self._list_locked()
            self._require_locked(entries, adapter_id)
            return self._lineage_locked(entries, adapter_id)

    def validate_continuation(
        self,
        *,
        adapter_id: str,
        base_model: str,
        base_model_revision: str | None,
        parent_adapter_id: str | None,
        parent_adapter_hash: str | None,
    ) -> None:
        with self._locked(exclusive=False):
            self._validate_lineage_locked(
                self._list_locked(),
                adapter_id=adapter_id,
                base_model=base_model,
                base_model_revision=base_model_revision,
                parent_adapter_id=parent_adapter_id,
                parent_adapter_hash=parent_adapter_hash,
            )

    def apply_evaluation(
        self,
        adapter_id: str,
        *,
        score: float,
        result_path: str | Path,
        next_status: AdapterStatus | None = None,
        holdout_score: float | None = None,
        holdout_baseline_score: float | None = None,
        drift_scores: dict[str, float] | None = None,
        activation_gate_passed: bool | None = None,
    ) -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            if entry.status != AdapterStatus.CANDIDATE:
                raise ValueError(
                    "Only candidate adapters can receive evaluation gating"
                )
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
                holdout_score=holdout_score,
                holdout_baseline_score=holdout_baseline_score,
                holdout_regression=(
                    holdout_score is not None
                    and holdout_baseline_score is not None
                    and holdout_score < holdout_baseline_score
                ),
                drift_scores=drift_scores,
                activation_gate_passed=(
                    next_status == AdapterStatus.TRIAL_ACTIVE
                    if activation_gate_passed is None
                    else activation_gate_passed
                ),
                rollout_state="shadow"
                if next_status == AdapterStatus.TRIAL_ACTIVE
                else next_status.value,
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

    def apply_behavioral_evaluation(
        self,
        adapter_id: str,
        *,
        evaluation_id: str,
        result_path: str | Path,
        gate_passed: bool,
    ) -> AdapterEntry:
        """Bind a behavioral result to the candidate activation gate."""

        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            if entry.status not in {AdapterStatus.CANDIDATE, AdapterStatus.TRIAL_ACTIVE}:
                raise ValueError(
                    "Only candidate or trial adapters can receive behavioral gating"
                )
            return self._replace_locked(
                entries,
                adapter_id,
                behavioral_evaluation_id=evaluation_id,
                behavioral_evaluation_path=str(result_path),
                behavioral_gate_passed=gate_passed,
                activation_gate_passed=(
                    entry.status == AdapterStatus.TRIAL_ACTIVE and gate_passed
                ),
            )

    def activate(
        self, adapter_id: str, *, activation_sequence: int | None = None
    ) -> AdapterEntry:
        with self._locked(exclusive=True):
            current_entries = self._list_locked()
            entry = self._require_locked(current_entries, adapter_id)
            self._ensure_transition(entry.status, AdapterStatus.ACTIVE)
            if not entry.activation_gate_passed:
                raise ValueError("Adapter evaluation regression gate has not passed")
            previous_active = next(
                (
                    item.adapter_id
                    for item in current_entries
                    if item.status == AdapterStatus.ACTIVE
                ),
                None,
            )
            entries = []
            activated: AdapterEntry | None = None
            now = _now_iso()
            for existing in current_entries:
                if existing.adapter_id == adapter_id:
                    activated = _copy_entry(
                        existing,
                        status=AdapterStatus.ACTIVE,
                        updated_at=now,
                        activation_sequence=activation_sequence,
                        rollout_state="canary",
                        rollback_target_id=previous_active,
                    )
                    entries.append(activated)
                elif existing.status == AdapterStatus.ACTIVE:
                    entries.append(
                        _copy_entry(
                            existing, status=AdapterStatus.ARCHIVED, updated_at=now
                        )
                    )
                else:
                    entries.append(existing)
            assert activated is not None
            self._write_locked(entries)
            return activated

    def restore_active(
        self, adapter_id: str | None, *, activation_sequence: int
    ) -> AdapterEntry | None:
        with self._locked(exclusive=True):
            current_entries = self._list_locked()
            target = (
                None
                if adapter_id is None
                else self._require_locked(current_entries, adapter_id)
            )
            if target is not None and target.status not in {
                AdapterStatus.ARCHIVED,
                AdapterStatus.APPROVED,
            }:
                raise ValueError("Rollback target is not archived or approved")
            restored: AdapterEntry | None = None
            entries = []
            now = _now_iso()
            for entry in current_entries:
                if entry.adapter_id == adapter_id:
                    restored = _copy_entry(
                        entry,
                        status=AdapterStatus.ACTIVE,
                        updated_at=now,
                        activation_sequence=activation_sequence,
                        rollout_state="stable",
                    )
                    entries.append(restored)
                elif entry.status == AdapterStatus.ACTIVE:
                    entries.append(
                        _copy_entry(
                            entry,
                            status=AdapterStatus.ARCHIVED,
                            updated_at=now,
                            rollout_state="rolled_back",
                        )
                    )
                else:
                    entries.append(entry)
            self._write_locked(entries)
            return restored

    def transition(self, adapter_id: str, status: AdapterStatus) -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            self._ensure_transition(entry.status, status)
            return self._replace_locked(entries, adapter_id, status=status)

    def record_canary(self, adapter_id: str, *, success: bool) -> AdapterEntry:
        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            if entry.status != AdapterStatus.ACTIVE or entry.rollout_state not in {
                "canary",
                "canary_failed",
            }:
                raise ValueError(
                    "Only an active canary adapter can receive canary results"
                )
            return self._replace_locked(
                entries,
                adapter_id,
                rollout_state=(
                    "stable"
                    if success
                    else "canary_failed"
                    if entry.canary_failures + 1
                    >= self.settings.adapter_registry.canary_failure_limit
                    else "canary"
                ),
                canary_failures=entry.canary_failures + (0 if success else 1),
            )

    def _lookup_locked(
        self, entries: builtins.list[AdapterEntry], adapter_id: str
    ) -> AdapterEntry | None:
        return next(
            (entry for entry in entries if entry.adapter_id == adapter_id), None
        )

    def _require_locked(
        self, entries: builtins.list[AdapterEntry], adapter_id: str
    ) -> AdapterEntry:
        entry = self._lookup_locked(entries, adapter_id)
        if entry is None:
            raise ValueError(f"Unknown adapter: {adapter_id}")
        return entry

    def _lineage_locked(
        self, entries: builtins.list[AdapterEntry], adapter_id: str
    ) -> builtins.list[AdapterEntry]:
        lineage: builtins.list[AdapterEntry] = []
        seen: set[str] = set()
        current_id: str | None = adapter_id
        while current_id is not None:
            if current_id in seen:
                raise ValueError(f"Cyclic adapter lineage detected at: {current_id}")
            seen.add(current_id)
            current = self._lookup_locked(entries, current_id)
            if current is None:
                raise ValueError(f"Unknown adapter in lineage: {current_id}")
            lineage.append(current)
            current_id = current.parent_adapter_id
        return lineage

    def _validate_lineage_locked(
        self,
        entries: builtins.list[AdapterEntry],
        *,
        adapter_id: str,
        base_model: str,
        base_model_revision: str | None,
        parent_adapter_id: str | None,
        parent_adapter_hash: str | None,
    ) -> None:
        if (parent_adapter_id is None) != (parent_adapter_hash is None):
            raise ValueError("Parent adapter ID and hash must be provided together")
        if parent_adapter_id is None:
            return
        if parent_adapter_id == adapter_id:
            raise ValueError("Cyclic adapter lineage is not allowed")
        parent = self._lookup_locked(entries, parent_adapter_id)
        if parent is None:
            raise ValueError(f"Unknown parent adapter: {parent_adapter_id}")
        if parent.adapter_hash != parent_adapter_hash:
            raise ValueError("Parent adapter hash mismatch")
        if parent.base_model != base_model:
            raise ValueError("Parent adapter base model mismatch")
        if parent.base_model_revision != base_model_revision:
            raise ValueError("Parent adapter base revision mismatch")
        if any(
            item.adapter_id == adapter_id
            for item in self._lineage_locked(entries, parent_adapter_id)
        ):
            raise ValueError("Cyclic adapter lineage is not allowed")

    def _replace_locked(
        self,
        current_entries: builtins.list[AdapterEntry],
        adapter_id: str,
        **updates: Any,
    ) -> AdapterEntry:
        entries: builtins.list[AdapterEntry] = []
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

    def _write_locked(self, entries: builtins.list[AdapterEntry]) -> None:
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
            AdapterStatus.CANDIDATE: {
                AdapterStatus.TRIAL_ACTIVE,
                AdapterStatus.REJECTED,
            },
            AdapterStatus.TRIAL_ACTIVE: {AdapterStatus.APPROVED},
            AdapterStatus.APPROVED: {AdapterStatus.ACTIVE},
            AdapterStatus.ACTIVE: {AdapterStatus.ARCHIVED},
            AdapterStatus.REJECTED: set(),
            AdapterStatus.ARCHIVED: set(),
        }
        if target not in allowed[current]:
            raise ValueError(
                f"Invalid adapter status transition: {current} -> {target}"
            )


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


def _dataset_record_hashes(path: Path) -> tuple[tuple[str, ...], int]:
    if not path.is_file():
        return (), 0
    hashes: list[str] = []
    try:
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            normalized = json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            hashes.append(hashlib.sha256(normalized.encode()).hexdigest())
    except (OSError, json.JSONDecodeError):
        return (), 0
    return tuple(hashes), len(hashes) - len(set(hashes))

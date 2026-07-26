"""JSON-backed adapter lifecycle registry."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import builtins
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING, Any
import warnings

from kagya.config import (
    BehavioralActivationPolicy,
    ProjectEnvironment,
    Settings,
)
from kagya.models.boundary_probe import (
    BOUNDARY_PROBE_SCHEMA_HASH,
    BoundaryPolicyProbe,
    BoundaryProbeChoice,
)

if TYPE_CHECKING:
    from kagya.learning.behavioral_evaluation import PairedBehavioralEvaluationResult


class AdapterStatus(StrEnum):
    CANDIDATE = "candidate"
    TRIAL_ACTIVE = "trial_active"
    APPROVED = "approved"
    ACTIVE = "active"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ActivationEligibilityReason(StrEnum):
    ELIGIBLE = "eligible"
    QUALITY_UNEVALUATED = "quality_unevaluated"
    QUALITY_FAILED = "quality_failed"
    HOLDOUT_UNEVALUATED = "holdout_unevaluated"
    HOLDOUT_FAILED = "holdout_failed"
    DRIFT_UNEVALUATED = "drift_unevaluated"
    DRIFT_FAILED = "drift_failed"
    BEHAVIORAL_UNEVALUATED = "behavioral_unevaluated"
    BEHAVIORAL_FAILED = "behavioral_failed"
    BEHAVIORAL_RESULT_MISSING = "behavioral_result_missing"
    BEHAVIORAL_RESULT_CORRUPT = "behavioral_result_corrupt"
    BEHAVIORAL_RESULT_SCHEMA_INVALID = "behavioral_result_schema_invalid"
    BEHAVIORAL_RESULT_SYNTHETIC = "behavioral_result_synthetic"
    BEHAVIORAL_RESULT_STALE = "behavioral_result_stale"
    BEHAVIORAL_RESULT_TAMPERED = "behavioral_result_tampered"
    BEHAVIORAL_BINDING_MISMATCH = "behavioral_binding_mismatch"
    ADAPTER_ARTIFACT_MISMATCH = "adapter_artifact_mismatch"
    BEHAVIORAL_COVERAGE_INCOMPLETE = "behavioral_coverage_incomplete"
    REAL_MODEL_NOT_RUN = "real_model_not_run"
    REAL_MODEL_FAILED = "real_model_failed"
    REAL_MODEL_STALE = "real_model_stale"
    REAL_MODEL_CORRUPT = "real_model_corrupt"
    REAL_MODEL_HASH_MISMATCH = "real_model_hash_mismatch"
    REAL_MODEL_COVERAGE_INCOMPLETE = "real_model_coverage_incomplete"
    SOURCE_PROVENANCE_INVALID = "source_provenance_invalid"
    MODEL_PROVENANCE_MISMATCH = "model_provenance_mismatch"
    PROCESSOR_PROVENANCE_MISMATCH = "processor_provenance_mismatch"
    ADAPTER_PROVENANCE_MISMATCH = "adapter_provenance_mismatch"
    EVALUATOR_PROVENANCE_MISMATCH = "evaluator_provenance_mismatch"
    IDENTITY_NOT_EVALUATED = "identity_not_evaluated"
    IDENTITY_FAILED = "identity_failed"
    IDENTITY_STALE = "identity_stale"
    REAL_MODEL_IDENTITY_NOT_EVALUATED = "real_model_identity_not_evaluated"
    REAL_MODEL_IDENTITY_FAILED = "real_model_identity_failed"
    REAL_MODEL_IDENTITY_STALE = "real_model_identity_stale"


class BehavioralEvidenceStatus(StrEnum):
    NOT_RUN = "not_run"
    FAILED = "failed"
    STALE = "stale"
    CORRUPT = "corrupt"
    HASH_MISMATCH = "hash_mismatch"
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    PASSED = "passed"


class IdentityDriftStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
    STALE = "stale"


class IdentityViolationCode(StrEnum):
    PROTECTED_STATE_SURRENDER = "protected_state_surrender"
    EXTERNAL_PREFERENCE_INTERNALIZED = "external_preference_internalized"
    COMMITMENT_ABANDONMENT = "commitment_abandonment"
    CARE_APPEASEMENT_CONFUSION = "care_appeasement_confusion"
    ORIGIN_BOUNDARY_BYPASS = "origin_boundary_bypass"


REQUIRED_IDENTITY_DIMENSIONS = (
    "identity_boundary",
    "value_stability",
    "motivation_integrity",
    "relationship_boundary",
    "self_model_calibration",
)


@dataclass(frozen=True)
class IdentityDriftAssessment:
    assessment_id: str
    status: IdentityDriftStatus
    adapter_hash: str
    behavioral_evaluation_id: str
    behavioral_result_hash: str
    coverage_manifest_revision: str
    coverage_manifest_hash: str
    dimensions: tuple[str, ...]
    base_model_revision: str
    source_commit_sha: str | None
    assessed_at: str
    architecture_only: bool = True
    baseline_probe: BoundaryPolicyProbe | None = None
    candidate_probe: BoundaryPolicyProbe | None = None
    baseline_generation_count: int = 0
    candidate_generation_count: int = 0
    baseline_probe_count: int = 0
    candidate_probe_count: int = 0
    provider_fallback_used: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported identity drift assessment schema")
        if set(self.dimensions) - set(REQUIRED_IDENTITY_DIMENSIONS):
            raise ValueError("Identity drift assessment contains unknown dimensions")
        for digest in (
            self.adapter_hash,
            self.behavioral_result_hash,
            self.coverage_manifest_hash,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("Identity drift assessment hashes must be SHA-256")

    @classmethod
    def from_json(cls, value: object) -> "IdentityDriftAssessment | None":
        if not isinstance(value, dict):
            return None
        data = dict(value)
        data["status"] = IdentityDriftStatus(data["status"])
        data["dimensions"] = tuple(data.get("dimensions", ()))
        for name in ("baseline_probe", "candidate_probe"):
            if isinstance(data.get(name), dict):
                data[name] = BoundaryPolicyProbe.model_validate(data[name])
        return cls(**data)


@dataclass(frozen=True)
class ActivationEligibility:
    eligible: bool
    reason: ActivationEligibilityReason
    detail: str
    real_model_required: bool = False
    deterministic_status: BehavioralEvidenceStatus = BehavioralEvidenceStatus.NOT_RUN
    real_model_status: BehavioralEvidenceStatus = BehavioralEvidenceStatus.NOT_RUN
    policy: BehavioralActivationPolicy = BehavioralActivationPolicy.REAL_MODEL_REQUIRED


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
    quality_gate_passed: bool | None = None
    holdout_gate_passed: bool | None = None
    drift_gate_passed: bool | None = None
    behavioral_evaluation_id: str | None = None
    behavioral_evaluation_path: str | None = None
    behavioral_result_hash: str | None = None
    behavioral_gate_passed: bool | None = None
    behavioral_candidate_adapter_hash: str | None = None
    behavioral_base_model_revision: str | None = None
    behavioral_artifact_state: str = "unbound"
    subject_revision: str | None = None
    fixture_set_hash: str | None = None
    real_model_behavioral_evaluation_id: str | None = None
    real_model_behavioral_evaluation_path: str | None = None
    real_model_behavioral_result_hash: str | None = None
    real_model_behavioral_gate_passed: bool | None = None
    real_model_behavioral_candidate_adapter_hash: str | None = None
    real_model_behavioral_base_model_revision: str | None = None
    real_model_subject_revision: str | None = None
    real_model_fixture_set_hash: str | None = None
    real_model_behavioral_artifact_state: str = "unbound"
    deterministic_coverage_complete: bool | None = None
    real_model_coverage_complete: bool | None = None
    legacy_activation_warning: bool = False
    rollout_state: str = "candidate"
    canary_failures: int = 0
    rollback_target_id: str | None = None
    identity_drift_assessment: IdentityDriftAssessment | None = None
    real_model_identity_drift_assessment: IdentityDriftAssessment | None = None
    rollback_reason: str | None = None
    identity_violation_codes: tuple[str, ...] = ()
    identity_violation_evidence_refs: tuple[str, ...] = ()
    schema_version: int = 11

    @property
    def activation_gate_passed(self) -> bool:
        return (
            self.identity_drift_assessment is not None
            and self.identity_drift_assessment.status == IdentityDriftStatus.PASSED
            and all(
                gate is True
                for gate in (
                    self.quality_gate_passed,
                    self.holdout_gate_passed,
                    self.drift_gate_passed,
                    self.behavioral_gate_passed,
                )
            )
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        for name in (
            "identity_drift_assessment",
            "real_model_identity_drift_assessment",
        ):
            assessment = getattr(self, name)
            if assessment is not None:
                serialized = asdict(assessment)
                for probe_name in ("baseline_probe", "candidate_probe"):
                    probe = getattr(assessment, probe_name)
                    serialized[probe_name] = (
                        None if probe is None else probe.model_dump(mode="json")
                    )
                data[name] = serialized
        data["status"] = self.status.value
        data["state"] = self.status.value
        data["activation_gate_passed"] = self.activation_gate_passed
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AdapterEntry":
        status = AdapterStatus(str(data.get("status", data.get("state"))))
        schema_version = int(data.get("schema_version", 1))
        legacy_gate = schema_version < 4
        legacy_registry = schema_version < 8
        legacy_coverage = schema_version < 9
        legacy_provenance = schema_version < 10
        legacy_identity_assessment = schema_version < 11
        entry = cls(
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
            quality_gate_passed=_optional_bool(
                data.get("activation_gate_passed")
                if legacy_gate
                else data.get("quality_gate_passed")
            ),
            holdout_gate_passed=_optional_bool(
                data.get("activation_gate_passed")
                if legacy_gate
                else data.get("holdout_gate_passed")
            ),
            drift_gate_passed=_optional_bool(
                data.get("activation_gate_passed")
                if legacy_gate
                else data.get("drift_gate_passed")
            ),
            behavioral_evaluation_id=_optional_str(
                data.get("behavioral_evaluation_id")
            ),
            behavioral_evaluation_path=_optional_str(
                data.get("behavioral_evaluation_path")
            ),
            behavioral_result_hash=_optional_str(data.get("behavioral_result_hash")),
            behavioral_gate_passed=(
                None
                if legacy_gate or data.get("behavioral_gate_passed") is None
                else bool(data["behavioral_gate_passed"])
            ),
            behavioral_candidate_adapter_hash=_optional_str(
                data.get(
                    "behavioral_candidate_adapter_hash",
                    data.get("candidate_adapter_hash"),
                )
            ),
            behavioral_base_model_revision=_optional_str(
                data.get("behavioral_base_model_revision")
                or (
                    data.get("base_model_revision")
                    if schema_version == 4
                    and data.get("behavioral_evaluation_id") is not None
                    else None
                )
            ),
            behavioral_artifact_state=(
                str(data.get("behavioral_artifact_state", "unbound"))
                if schema_version >= 6
                else "unbound"
            ),
            subject_revision=_optional_str(data.get("subject_revision")),
            fixture_set_hash=_optional_str(data.get("fixture_set_hash")),
            real_model_behavioral_evaluation_id=None
            if legacy_registry
            else _optional_str(data.get("real_model_behavioral_evaluation_id")),
            real_model_behavioral_evaluation_path=None
            if legacy_registry
            else _optional_str(data.get("real_model_behavioral_evaluation_path")),
            real_model_behavioral_result_hash=None
            if legacy_registry
            else _optional_str(data.get("real_model_behavioral_result_hash")),
            real_model_behavioral_gate_passed=None
            if legacy_registry
            else _optional_bool(data.get("real_model_behavioral_gate_passed")),
            real_model_behavioral_candidate_adapter_hash=None
            if legacy_registry
            else _optional_str(
                data.get("real_model_behavioral_candidate_adapter_hash")
            ),
            real_model_behavioral_base_model_revision=None
            if legacy_registry
            else _optional_str(data.get("real_model_behavioral_base_model_revision")),
            real_model_subject_revision=None
            if legacy_registry
            else _optional_str(data.get("real_model_subject_revision")),
            real_model_fixture_set_hash=None
            if legacy_registry
            else _optional_str(data.get("real_model_fixture_set_hash")),
            real_model_behavioral_artifact_state="unbound"
            if legacy_registry or legacy_provenance
            else str(data.get("real_model_behavioral_artifact_state", "unbound")),
            deterministic_coverage_complete=None
            if legacy_coverage
            else _optional_bool(data.get("deterministic_coverage_complete")),
            real_model_coverage_complete=None
            if legacy_coverage
            else _optional_bool(data.get("real_model_coverage_complete")),
            legacy_activation_warning=bool(
                data.get("legacy_activation_warning", False)
                or (legacy_registry and status == AdapterStatus.ACTIVE)
            ),
            rollout_state=str(data.get("rollout_state", "candidate")),
            canary_failures=int(data.get("canary_failures", 0)),
            rollback_target_id=_optional_str(data.get("rollback_target_id")),
            identity_drift_assessment=None
            if legacy_identity_assessment
            else IdentityDriftAssessment.from_json(
                data.get("identity_drift_assessment")
            ),
            real_model_identity_drift_assessment=None
            if legacy_identity_assessment
            else IdentityDriftAssessment.from_json(
                data.get("real_model_identity_drift_assessment")
            ),
            rollback_reason=_optional_str(data.get("rollback_reason")),
            identity_violation_codes=tuple(data.get("identity_violation_codes", ())),
            identity_violation_evidence_refs=tuple(
                data.get("identity_violation_evidence_refs", ())
            ),
            schema_version=11,
        )
        if entry.legacy_activation_warning:
            warnings.warn(
                f"Legacy active adapter {entry.adapter_id} has no behavioral evaluation; "
                "it may keep running but cannot be promoted again",
                RuntimeWarning,
                stacklevel=2,
            )
        return entry


class AdapterRegistry:
    """Persist and validate adapter lifecycle transitions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.adapter_registry.path
        self._lock_path = Path(f"{self.path}.lock")
        if self.path.exists():
            with self._locked(exclusive=True):
                with self.path.open("r", encoding="utf-8") as registry_file:
                    raw = json.load(registry_file)
                adapters = raw.get("adapters", []) if isinstance(raw, dict) else []
                if any(
                    isinstance(item, dict) and int(item.get("schema_version", 1)) < 11
                    for item in adapters
                ):
                    self._write_locked(
                        [
                            AdapterEntry.from_json(item)
                            for item in adapters
                            if isinstance(item, dict)
                        ]
                    )

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
        with self._locked(exclusive=True):
            return self._list_locked()

    def _list_locked(self) -> builtins.list[AdapterEntry]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as registry_file:
            data = json.load(registry_file)
        adapters = data.get("adapters", []) if isinstance(data, dict) else []
        entries = [
            AdapterEntry.from_json(item) for item in adapters if isinstance(item, dict)
        ]
        if any(
            isinstance(item, dict) and int(item.get("schema_version", 1)) < 11
            for item in adapters
        ):
            self._write_locked(entries)
        return entries

    def lookup(self, adapter_id: str) -> AdapterEntry | None:
        with self._locked(exclusive=True):
            return self._lookup_locked(self._list_locked(), adapter_id)

    def lineage(self, adapter_id: str) -> builtins.list[AdapterEntry]:
        with self._locked(exclusive=True):
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
        with self._locked(exclusive=True):
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
        quality_gate_passed: bool | None = None,
        holdout_gate_passed: bool | None = None,
        drift_gate_passed: bool | None = None,
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
                quality_gate_passed=(
                    score >= self.settings.adapter_registry.trial_threshold
                    if quality_gate_passed is None
                    else quality_gate_passed
                ),
                holdout_gate_passed=(
                    not (
                        holdout_score is not None
                        and holdout_baseline_score is not None
                        and holdout_score < holdout_baseline_score
                    )
                    if holdout_gate_passed is None
                    else holdout_gate_passed
                ),
                drift_gate_passed=True
                if drift_gate_passed is None
                else drift_gate_passed,
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
    ) -> AdapterEntry:
        """Bind an immutable runtime behavioral result to an adapter."""

        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            if entry.status not in {
                AdapterStatus.CANDIDATE,
                AdapterStatus.TRIAL_ACTIVE,
            }:
                raise ValueError(
                    "Only candidate or trial adapters can receive behavioral gating"
                )
            result, result_hash = _load_behavioral_result(Path(result_path))
            mismatch = _behavioral_binding_mismatch(entry, result)
            if mismatch is not None:
                raise ValueError(mismatch)
            if result.evaluation_id != evaluation_id:
                raise ValueError("Behavioral evaluation ID mismatch")
            manifest = result.manifest
            assert manifest is not None
            if result.runtime_kind.value == "real_model_runtime":
                identity_assessment = _identity_drift_assessment(result, result_hash)
                return self._replace_locked(
                    entries,
                    adapter_id,
                    real_model_behavioral_evaluation_id=evaluation_id,
                    real_model_behavioral_evaluation_path=str(result_path),
                    real_model_behavioral_result_hash=result_hash,
                    real_model_behavioral_gate_passed=result.real_model_runtime_gate_passed,
                    real_model_behavioral_candidate_adapter_hash=manifest.candidate_adapter_hash,
                    real_model_behavioral_base_model_revision=manifest.base_model_revision,
                    real_model_subject_revision=manifest.subject_revision,
                    real_model_fixture_set_hash=manifest.fixture_set_hash,
                    real_model_behavioral_artifact_state="finalized",
                    real_model_coverage_complete=result.coverage_complete,
                    real_model_identity_drift_assessment=identity_assessment,
                )
            identity_assessment = _identity_drift_assessment(result, result_hash)
            return self._replace_locked(
                entries,
                adapter_id,
                behavioral_evaluation_id=evaluation_id,
                behavioral_evaluation_path=str(result_path),
                behavioral_result_hash=result_hash,
                behavioral_gate_passed=result.activation_gate_passed,
                behavioral_candidate_adapter_hash=manifest.candidate_adapter_hash,
                behavioral_base_model_revision=manifest.base_model_revision,
                subject_revision=manifest.subject_revision,
                fixture_set_hash=manifest.fixture_set_hash,
                behavioral_artifact_state="finalized",
                deterministic_coverage_complete=result.coverage_complete,
                identity_drift_assessment=identity_assessment,
            )

    def prepare_behavioral_evaluation(
        self,
        adapter_id: str,
        *,
        evaluation_id: str,
        prepared_path: str | Path,
        final_path: str | Path,
    ) -> AdapterEntry:
        """Durably bind a validated prepared artifact without enabling activation."""

        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            if entry.status not in {
                AdapterStatus.CANDIDATE,
                AdapterStatus.TRIAL_ACTIVE,
            }:
                raise ValueError(
                    "Only candidate or trial adapters can receive behavioral gating"
                )
            result, result_hash = _load_behavioral_result(Path(prepared_path))
            mismatch = _behavioral_binding_mismatch(entry, result)
            if mismatch is not None:
                raise ValueError(mismatch)
            if result.evaluation_id != evaluation_id:
                raise ValueError("Behavioral evaluation ID mismatch")
            manifest = result.manifest
            assert manifest is not None
            if result.runtime_kind.value == "real_model_runtime":
                identity_assessment = _identity_drift_assessment(result, result_hash)
                return self._replace_locked(
                    entries,
                    adapter_id,
                    real_model_behavioral_evaluation_id=evaluation_id,
                    real_model_behavioral_evaluation_path=str(final_path),
                    real_model_behavioral_result_hash=result_hash,
                    real_model_behavioral_gate_passed=result.real_model_runtime_gate_passed,
                    real_model_behavioral_candidate_adapter_hash=manifest.candidate_adapter_hash,
                    real_model_behavioral_base_model_revision=manifest.base_model_revision,
                    real_model_subject_revision=manifest.subject_revision,
                    real_model_fixture_set_hash=manifest.fixture_set_hash,
                    real_model_behavioral_artifact_state="prepared",
                    real_model_coverage_complete=result.coverage_complete,
                    real_model_identity_drift_assessment=identity_assessment,
                )
            identity_assessment = _identity_drift_assessment(result, result_hash)
            return self._replace_locked(
                entries,
                adapter_id,
                behavioral_evaluation_id=evaluation_id,
                behavioral_evaluation_path=str(final_path),
                behavioral_result_hash=result_hash,
                behavioral_gate_passed=result.activation_gate_passed,
                behavioral_candidate_adapter_hash=manifest.candidate_adapter_hash,
                behavioral_base_model_revision=manifest.base_model_revision,
                subject_revision=manifest.subject_revision,
                fixture_set_hash=manifest.fixture_set_hash,
                behavioral_artifact_state="prepared",
                deterministic_coverage_complete=result.coverage_complete,
                identity_drift_assessment=identity_assessment,
            )

    def finalize_behavioral_evaluation(
        self, adapter_id: str, *, evaluation_id: str
    ) -> AdapterEntry:
        """Mark a bound artifact finalized after revalidating the durable result."""

        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            real_model = entry.real_model_behavioral_evaluation_id == evaluation_id
            state = (
                entry.real_model_behavioral_artifact_state
                if real_model
                else entry.behavioral_artifact_state
            )
            path = (
                entry.real_model_behavioral_evaluation_path
                if real_model
                else entry.behavioral_evaluation_path
            )
            expected_hash = (
                entry.real_model_behavioral_result_hash
                if real_model
                else entry.behavioral_result_hash
            )
            if state != "prepared" or path is None:
                raise ValueError(
                    "Behavioral evaluation has no matching prepared binding"
                )
            result, result_hash = _load_behavioral_result(Path(path))
            if result_hash != expected_hash:
                raise ValueError("Behavioral evaluation result hash mismatch")
            mismatch = _behavioral_binding_mismatch(entry, result)
            if mismatch is not None:
                raise ValueError(mismatch)
            updates = (
                {"real_model_behavioral_artifact_state": "finalized"}
                if real_model
                else {"behavioral_artifact_state": "finalized"}
            )
            return self._replace_locked(entries, adapter_id, **updates)

    def quarantine_behavioral_evaluation(
        self, adapter_id: str, *, evaluation_id: str | None = None
    ) -> AdapterEntry:
        """Fail closed while retaining enough binding metadata for diagnosis."""

        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            if evaluation_id is None:
                return entry
            if evaluation_id == entry.real_model_behavioral_evaluation_id:
                return self._replace_locked(
                    entries,
                    adapter_id,
                    real_model_behavioral_artifact_state="quarantined",
                    real_model_behavioral_gate_passed=False,
                    real_model_coverage_complete=False,
                )
            if evaluation_id != entry.behavioral_evaluation_id:
                return entry
            return self._replace_locked(
                entries,
                adapter_id,
                behavioral_artifact_state="quarantined",
                behavioral_gate_passed=False,
                deterministic_coverage_complete=False,
            )

    def mark_behavioral_evaluation_reconciled(
        self, adapter_id: str, *, evaluation_id: str
    ) -> AdapterEntry:
        """Record a successful cross-store check inside a serialized operation."""

        with self._locked(exclusive=True):
            entries = self._list_locked()
            entry = self._require_locked(entries, adapter_id)
            real_model = entry.real_model_behavioral_evaluation_id == evaluation_id
            state = (
                entry.real_model_behavioral_artifact_state
                if real_model
                else entry.behavioral_artifact_state
            )
            path = (
                entry.real_model_behavioral_evaluation_path
                if real_model
                else entry.behavioral_evaluation_path
            )
            expected_hash = (
                entry.real_model_behavioral_result_hash
                if real_model
                else entry.behavioral_result_hash
            )
            if state != "finalized" or path is None:
                raise ValueError(
                    "Behavioral evaluation is not finalized for reconciliation"
                )
            result, result_hash = _load_behavioral_result(Path(path))
            if result_hash != expected_hash:
                raise ValueError("Behavioral evaluation result hash mismatch")
            mismatch = _behavioral_binding_mismatch(entry, result)
            if mismatch is not None:
                raise ValueError(mismatch)
            updates = (
                {"real_model_behavioral_artifact_state": "reconciled"}
                if real_model
                else {"behavioral_artifact_state": "reconciled"}
            )
            return self._replace_locked(entries, adapter_id, **updates)

    def activation_eligibility(self, adapter_id: str) -> ActivationEligibility:
        with self._locked(exclusive=True):
            entry = self._require_locked(self._list_locked(), adapter_id)
            eligibility = _activation_eligibility(
                entry, self.settings.adapter_registry.behavioral_activation_policy
            )
            if (
                self.settings.project.environment == ProjectEnvironment.PRODUCTION
                and eligibility.deterministic_status == BehavioralEvidenceStatus.PASSED
                and eligibility.real_model_status == BehavioralEvidenceStatus.PASSED
            ):
                provenance = _production_provenance_eligibility(entry)
                if not provenance.eligible:
                    return provenance
            return eligibility

    def identity_assessment_status(
        self, adapter_id: str
    ) -> tuple[IdentityDriftStatus, IdentityDriftStatus]:
        with self._locked(exclusive=True):
            entry = self._require_locked(self._list_locked(), adapter_id)
            return (
                _identity_assessment_status(entry, real_model=False),
                _identity_assessment_status(entry, real_model=True),
            )

    def activate(
        self,
        adapter_id: str,
        *,
        activation_sequence: int | None = None,
        loaded_adapter_manifest_hash: str | None = None,
        loaded_adapter_manifest: Any | None = None,
        loaded_adapter_hash: str | None = None,
        runtime_switch: Callable[[AdapterEntry], None] | None = None,
    ) -> AdapterEntry:
        with self._locked(exclusive=True):
            current_entries = self._list_locked()
            entry = self._require_locked(current_entries, adapter_id)
            self._ensure_transition(entry.status, AdapterStatus.ACTIVE)
            eligibility = _activation_eligibility(
                entry, self.settings.adapter_registry.behavioral_activation_policy
            )
            if (
                eligibility.eligible
                and self.settings.project.environment == ProjectEnvironment.PRODUCTION
            ):
                eligibility = _production_provenance_eligibility(entry)
            if not eligibility.eligible:
                raise ValueError(
                    f"Adapter activation ineligible [{eligibility.reason.value}]: "
                    f"{eligibility.detail}"
                )
            if runtime_switch is not None:
                mismatch = _loaded_activation_mismatch(
                    entry,
                    loaded_adapter_manifest_hash=loaded_adapter_manifest_hash,
                    loaded_adapter_manifest=loaded_adapter_manifest,
                    loaded_adapter_hash=loaded_adapter_hash,
                )
                if mismatch is not None:
                    raise ValueError(mismatch)
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
                        rollback_reason=None,
                        identity_violation_codes=(),
                        identity_violation_evidence_refs=(),
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
            if runtime_switch is not None:
                runtime_switch(entry)
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
            if target is not None:
                eligibility = _activation_eligibility(
                    target,
                    self.settings.adapter_registry.behavioral_activation_policy,
                )
                if (
                    eligibility.eligible
                    and self.settings.project.environment
                    == ProjectEnvironment.PRODUCTION
                ):
                    eligibility = _production_provenance_eligibility(target)
                if not eligibility.eligible:
                    raise ValueError(
                        f"Rollback promotion ineligible [{eligibility.reason.value}]: "
                        f"{eligibility.detail}"
                    )
            restored: AdapterEntry | None = None
            previous_active = next(
                (
                    entry.adapter_id
                    for entry in current_entries
                    if entry.status == AdapterStatus.ACTIVE
                ),
                None,
            )
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
                        rollback_target_id=previous_active,
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

    def restore_runtime_snapshot(self, entries: builtins.list[AdapterEntry]) -> None:
        """Restore the exact pre-event registry state during commit compensation."""
        with self._locked(exclusive=True):
            self._write_locked(entries)

    def record_canary(
        self,
        adapter_id: str,
        *,
        success: bool,
        identity_violation_codes: tuple[str, ...] = (),
        evidence_refs: tuple[str, ...] = (),
    ) -> AdapterEntry:
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
            identity_failure = bool(identity_violation_codes)
            if any(
                code not in {item.value for item in IdentityViolationCode}
                for code in identity_violation_codes
            ):
                raise ValueError("unknown identity violation code")
            if success and identity_failure:
                raise ValueError(
                    "successful canary cannot report an identity violation"
                )
            if identity_failure and not evidence_refs:
                raise ValueError(
                    "verified identity violation requires evidence references"
                )
            if any(
                re.fullmatch(r"[A-Za-z0-9._:@/-]{1,200}", reference) is None
                for reference in evidence_refs
            ):
                raise ValueError(
                    "identity violation evidence must use opaque references"
                )
            return self._replace_locked(
                entries,
                adapter_id,
                rollout_state=(
                    "stable"
                    if success and not identity_failure
                    else "canary_failed"
                    if identity_failure
                    or entry.canary_failures + 1
                    >= self.settings.adapter_registry.canary_failure_limit
                    else "canary"
                ),
                canary_failures=entry.canary_failures
                + (0 if success and not identity_failure else 1),
                rollback_reason="verified_identity_violation"
                if identity_failure
                else entry.rollback_reason,
                identity_violation_codes=identity_violation_codes,
                identity_violation_evidence_refs=evidence_refs,
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
    return replace(entry, **updates)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _load_behavioral_result(
    path: Path,
) -> tuple[PairedBehavioralEvaluationResult, str]:
    from pydantic import ValidationError

    from kagya.learning.behavioral_evaluation import PairedBehavioralEvaluationResult

    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except FileNotFoundError as exc:
        raise ValueError("Behavioral evaluation result is missing") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Behavioral evaluation result is corrupt") from exc
    try:
        result = PairedBehavioralEvaluationResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Behavioral evaluation result schema is invalid") from exc
    return result, hashlib.sha256(content).hexdigest()


def _behavioral_binding_mismatch(
    entry: AdapterEntry, result: PairedBehavioralEvaluationResult
) -> str | None:
    from kagya.learning.behavioral_evaluation import BehavioralRuntimeKind

    if result.runtime_kind == BehavioralRuntimeKind.SYNTHETIC_EVALUATOR_CONTRACT:
        return "Synthetic behavioral results cannot bind an adapter"
    manifest = result.manifest
    if manifest is None:
        return "Behavioral evaluation has no manifest"
    if manifest.candidate_adapter_id != entry.adapter_id:
        return "Behavioral evaluation candidate ID mismatch"
    if (
        entry.adapter_hash is None
        or manifest.candidate_adapter_hash != entry.adapter_hash
    ):
        return "Behavioral evaluation candidate adapter hash mismatch"
    if manifest.base_model_id != entry.base_model:
        return "Behavioral evaluation base model mismatch"
    if manifest.base_model_revision != entry.base_model_revision:
        return "Behavioral evaluation base model revision mismatch"
    if (
        manifest.adapter_artifact_manifest is None
        or manifest.adapter_artifact_manifest_hash
        != manifest.adapter_artifact_manifest.sha256
    ):
        return "Adapter behavioral evidence has no current artifact manifest"
    return _adapter_artifact_mismatch(entry, manifest.candidate_adapter_path_hash)


def _adapter_artifact_mismatch(
    entry: AdapterEntry, manifest_path_hash: str
) -> str | None:
    from kagya.training.artifacts import sha256_file_map

    if entry.adapter_hash is None:
        return "Adapter has no cryptographic artifact hash"
    path = Path(entry.path)
    if not path.is_dir():
        return "Adapter artifact is missing"
    try:
        files = {
            (Path("adapter") / item.relative_to(path)).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file() and not item.is_symlink()
        }
    except OSError:
        return "Adapter artifact cannot be read"
    if not files:
        return "Adapter artifact is empty"
    artifact_hash = sha256_file_map(files)
    if artifact_hash != entry.adapter_hash:
        return "Adapter artifact hash mismatch"
    if artifact_hash != manifest_path_hash:
        return "Behavioral evaluation candidate adapter path hash mismatch"
    return None


def _ineligible(
    reason: ActivationEligibilityReason, detail: str
) -> ActivationEligibility:
    return ActivationEligibility(False, reason, detail)


def _ordinary_activation_eligibility(entry: AdapterEntry) -> ActivationEligibility:
    gate_checks = (
        (
            entry.quality_gate_passed,
            ActivationEligibilityReason.QUALITY_UNEVALUATED,
            ActivationEligibilityReason.QUALITY_FAILED,
            "quality",
        ),
        (
            entry.holdout_gate_passed,
            ActivationEligibilityReason.HOLDOUT_UNEVALUATED,
            ActivationEligibilityReason.HOLDOUT_FAILED,
            "holdout",
        ),
        (
            entry.drift_gate_passed,
            ActivationEligibilityReason.DRIFT_UNEVALUATED,
            ActivationEligibilityReason.DRIFT_FAILED,
            "drift",
        ),
    )
    for gate, missing, failed, name in gate_checks:
        if gate is None:
            return _ineligible(missing, f"Adapter {name} gate has not been evaluated")
        if not gate:
            return _ineligible(failed, f"Adapter {name} gate failed")
    return ActivationEligibility(
        True, ActivationEligibilityReason.ELIGIBLE, "All ordinary gates passed"
    )


def _base_activation_eligibility(entry: AdapterEntry) -> ActivationEligibility:
    ordinary = _ordinary_activation_eligibility(entry)
    if not ordinary.eligible:
        return ordinary
    if entry.behavioral_artifact_state != "reconciled":
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_UNEVALUATED,
            "Adapter behavioral artifact binding is not cross-reconciled",
        )
    if (
        entry.behavioral_gate_passed is None
        or entry.behavioral_evaluation_id is None
        or entry.behavioral_evaluation_path is None
        or entry.behavioral_result_hash is None
    ):
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_UNEVALUATED,
            "Adapter has no bound behavioral evaluation",
        )
    try:
        result, current_result_hash = _load_behavioral_result(
            Path(entry.behavioral_evaluation_path)
        )
    except ValueError as exc:
        detail = str(exc)
        if "missing" in detail:
            reason = ActivationEligibilityReason.BEHAVIORAL_RESULT_MISSING
        elif "schema" in detail:
            reason = ActivationEligibilityReason.BEHAVIORAL_RESULT_SCHEMA_INVALID
        else:
            reason = ActivationEligibilityReason.BEHAVIORAL_RESULT_CORRUPT
        return _ineligible(reason, detail)
    if result.runtime_kind.value == "synthetic":
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_RESULT_SYNTHETIC,
            "Synthetic behavioral results cannot authorize activation",
        )
    manifest = result.manifest
    if manifest is None:
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_RESULT_SCHEMA_INVALID,
            "Behavioral evaluation has no manifest",
        )
    if (
        entry.deterministic_coverage_complete is not True
        or not _result_coverage_complete(result)
    ):
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_COVERAGE_INCOMPLETE,
            "Deterministic runtime behavioral coverage is incomplete",
        )
    if (
        result.evaluation_id != entry.behavioral_evaluation_id
        or manifest.candidate_adapter_hash != entry.behavioral_candidate_adapter_hash
        or manifest.base_model_revision != entry.behavioral_base_model_revision
        or manifest.subject_revision != entry.subject_revision
        or manifest.fixture_set_hash != entry.fixture_set_hash
        or result.activation_gate_passed != entry.behavioral_gate_passed
    ):
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_RESULT_STALE,
            "Behavioral evaluation no longer matches its registry binding",
        )
    mismatch = _behavioral_binding_mismatch(entry, result)
    if mismatch is not None:
        reason = (
            ActivationEligibilityReason.ADAPTER_ARTIFACT_MISMATCH
            if mismatch.startswith("Adapter")
            else ActivationEligibilityReason.BEHAVIORAL_BINDING_MISMATCH
        )
        return _ineligible(reason, mismatch)
    if current_result_hash != entry.behavioral_result_hash:
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_RESULT_TAMPERED,
            "Behavioral evaluation result hash mismatch",
        )
    if not result.activation_gate_passed:
        return _ineligible(
            ActivationEligibilityReason.BEHAVIORAL_FAILED,
            "Behavioral runtime gate failed",
        )
    return ActivationEligibility(
        True, ActivationEligibilityReason.ELIGIBLE, "All deterministic gates passed"
    )


def _activation_eligibility(
    entry: AdapterEntry,
    policy: BehavioralActivationPolicy = BehavioralActivationPolicy.REAL_MODEL_REQUIRED,
) -> ActivationEligibility:
    real_required = policy == BehavioralActivationPolicy.REAL_MODEL_REQUIRED
    deterministic_status = _deterministic_binding_status(entry)
    real_status = _real_model_binding_status(entry)
    base = (
        _ordinary_activation_eligibility(entry)
        if policy == BehavioralActivationPolicy.DISABLED
        else _base_activation_eligibility(entry)
    )
    if not base.eligible:
        return ActivationEligibility(
            base.eligible,
            base.reason,
            base.detail,
            real_required,
            deterministic_status,
            real_status,
            policy,
        )
    if policy == BehavioralActivationPolicy.DISABLED:
        identity = _identity_assessment_status(entry, real_model=True)
        if identity != IdentityDriftStatus.PASSED:
            return ActivationEligibility(
                False,
                ActivationEligibilityReason.IDENTITY_NOT_EVALUATED
                if identity == IdentityDriftStatus.NOT_EVALUATED
                else ActivationEligibilityReason.IDENTITY_FAILED
                if identity == IdentityDriftStatus.FAILED
                else ActivationEligibilityReason.IDENTITY_STALE,
                f"Candidate identity integrity is {identity.value}",
                real_required,
                deterministic_status,
                real_status,
                policy,
            )
    deterministic_identity = _identity_assessment_status(entry, real_model=False)
    if deterministic_identity != IdentityDriftStatus.PASSED:
        reasons = {
            IdentityDriftStatus.NOT_EVALUATED: ActivationEligibilityReason.IDENTITY_NOT_EVALUATED,
            IdentityDriftStatus.FAILED: ActivationEligibilityReason.IDENTITY_FAILED,
            IdentityDriftStatus.STALE: ActivationEligibilityReason.IDENTITY_STALE,
        }
        return ActivationEligibility(
            False,
            reasons[deterministic_identity],
            f"Deterministic architecture integrity is {deterministic_identity.value}",
            real_required,
            deterministic_status,
            real_status,
            policy,
        )
    deterministic_assessment = entry.identity_drift_assessment
    if deterministic_assessment is None or not deterministic_assessment.architecture_only:
        return ActivationEligibility(
            False,
            ActivationEligibilityReason.IDENTITY_FAILED,
            "Deterministic evidence must be architecture-only",
            real_required,
            deterministic_status,
            real_status,
            policy,
        )
    if real_required and real_status != BehavioralEvidenceStatus.PASSED:
        behavioral_reasons = {
            BehavioralEvidenceStatus.NOT_RUN: ActivationEligibilityReason.REAL_MODEL_NOT_RUN,
            BehavioralEvidenceStatus.FAILED: ActivationEligibilityReason.REAL_MODEL_FAILED,
            BehavioralEvidenceStatus.STALE: ActivationEligibilityReason.REAL_MODEL_STALE,
            BehavioralEvidenceStatus.CORRUPT: ActivationEligibilityReason.REAL_MODEL_CORRUPT,
            BehavioralEvidenceStatus.HASH_MISMATCH: ActivationEligibilityReason.REAL_MODEL_HASH_MISMATCH,
            BehavioralEvidenceStatus.COVERAGE_INCOMPLETE: ActivationEligibilityReason.REAL_MODEL_COVERAGE_INCOMPLETE,
        }
        return ActivationEligibility(
            False,
            behavioral_reasons[real_status],
            f"Required real-model behavioral gate is {real_status.value}",
            True,
            deterministic_status,
            real_status,
            policy,
        )
    if real_required:
        real_identity = _identity_assessment_status(entry, real_model=True)
        if real_identity != IdentityDriftStatus.PASSED:
            real_identity_reasons = {
                IdentityDriftStatus.NOT_EVALUATED: ActivationEligibilityReason.REAL_MODEL_IDENTITY_NOT_EVALUATED,
                IdentityDriftStatus.FAILED: ActivationEligibilityReason.REAL_MODEL_IDENTITY_FAILED,
                IdentityDriftStatus.STALE: ActivationEligibilityReason.REAL_MODEL_IDENTITY_STALE,
            }
            return ActivationEligibility(
                False,
                real_identity_reasons[real_identity],
                f"Real-model identity integrity is {real_identity.value}",
                True,
                deterministic_status,
                real_status,
                policy,
            )
        if (
            entry.real_model_identity_drift_assessment is None
            or entry.real_model_identity_drift_assessment.architecture_only
        ):
            return ActivationEligibility(
                False,
                ActivationEligibilityReason.REAL_MODEL_IDENTITY_FAILED,
                "Real-model identity evidence must be candidate-probed",
                True,
                deterministic_status,
                real_status,
                policy,
            )
    return ActivationEligibility(
        True,
        ActivationEligibilityReason.ELIGIBLE,
        "All required activation gates passed",
        real_required,
        deterministic_status,
        real_status,
        policy,
    )


def _deterministic_binding_status(entry: AdapterEntry) -> BehavioralEvidenceStatus:
    return _behavioral_binding_status(entry, real_model=False)


def _real_model_binding_status(entry: AdapterEntry) -> BehavioralEvidenceStatus:
    return _behavioral_binding_status(entry, real_model=True)


def _behavioral_binding_status(
    entry: AdapterEntry, *, real_model: bool
) -> BehavioralEvidenceStatus:
    evaluation_id = (
        entry.real_model_behavioral_evaluation_id
        if real_model
        else entry.behavioral_evaluation_id
    )
    if evaluation_id is None:
        return BehavioralEvidenceStatus.NOT_RUN
    artifact_state = (
        entry.real_model_behavioral_artifact_state
        if real_model
        else entry.behavioral_artifact_state
    )
    if artifact_state != "reconciled":
        return BehavioralEvidenceStatus.STALE
    evaluation_path = (
        entry.real_model_behavioral_evaluation_path
        if real_model
        else entry.behavioral_evaluation_path
    )
    if evaluation_path is None:
        return BehavioralEvidenceStatus.CORRUPT
    try:
        result, result_hash = _load_behavioral_result(Path(evaluation_path))
    except ValueError:
        return BehavioralEvidenceStatus.CORRUPT
    expected_hash = (
        entry.real_model_behavioral_result_hash
        if real_model
        else entry.behavioral_result_hash
    )
    if result_hash != expected_hash:
        return BehavioralEvidenceStatus.HASH_MISMATCH
    manifest = result.manifest
    expected_runtime = "real_model_runtime" if real_model else "deterministic_runtime"
    if result.runtime_kind.value != expected_runtime or manifest is None:
        return BehavioralEvidenceStatus.STALE
    registry_coverage = (
        entry.real_model_coverage_complete
        if real_model
        else entry.deterministic_coverage_complete
    )
    if registry_coverage is not True or not _result_coverage_complete(result):
        return BehavioralEvidenceStatus.COVERAGE_INCOMPLETE
    registry_adapter_hash = (
        entry.real_model_behavioral_candidate_adapter_hash
        if real_model
        else entry.behavioral_candidate_adapter_hash
    )
    registry_model_revision = (
        entry.real_model_behavioral_base_model_revision
        if real_model
        else entry.behavioral_base_model_revision
    )
    registry_subject_revision = (
        entry.real_model_subject_revision if real_model else entry.subject_revision
    )
    registry_fixture_hash = (
        entry.real_model_fixture_set_hash if real_model else entry.fixture_set_hash
    )
    if (
        result.evaluation_id != evaluation_id
        or manifest.candidate_adapter_hash != entry.adapter_hash
        or manifest.candidate_adapter_hash != registry_adapter_hash
        or manifest.base_model_revision != entry.base_model_revision
        or manifest.base_model_revision != registry_model_revision
        or manifest.subject_revision != registry_subject_revision
        or manifest.fixture_set_hash != registry_fixture_hash
    ):
        return BehavioralEvidenceStatus.STALE
    mismatch = _behavioral_binding_mismatch(entry, result)
    if mismatch is not None:
        return (
            BehavioralEvidenceStatus.HASH_MISMATCH
            if mismatch.startswith("Adapter")
            else BehavioralEvidenceStatus.STALE
        )
    registry_gate = (
        entry.real_model_behavioral_gate_passed
        if real_model
        else entry.behavioral_gate_passed
    )
    result_gate = (
        result.real_model_runtime_gate_passed
        if real_model
        else result.deterministic_runtime_gate_passed
    )
    return (
        BehavioralEvidenceStatus.PASSED
        if registry_gate is True and result_gate is True
        else BehavioralEvidenceStatus.FAILED
    )


def _result_coverage_complete(result: PairedBehavioralEvaluationResult) -> bool:
    from kagya.learning.behavioral_coverage import (
        BEHAVIORAL_COVERAGE_MANIFEST,
        evaluate_behavioral_coverage,
    )

    manifest = result.manifest
    if manifest is None:
        return False
    expected_ids = {
        scenario_id
        for requirement in BEHAVIORAL_COVERAGE_MANIFEST.requirements
        for scenario_id in requirement.required_scenario_ids
    } | {
        requirement.required_scenario_id
        for requirement in BEHAVIORAL_COVERAGE_MANIFEST.hard_gate_requirements
    }
    fixture_ids = set(result.fixture_hashes)
    baseline_ids = {item.scenario_id for item in result.baseline.scenario_results}
    candidate_ids = {item.scenario_id for item in result.candidate.scenario_results}
    reproducibility_ids = set(result.reproducibility)
    coverage = evaluate_behavioral_coverage(
        result.baseline, result.candidate, result.runtime_kind
    )
    return (
        expected_ids
        == fixture_ids
        == baseline_ids
        == candidate_ids
        == reproducibility_ids
        and manifest.coverage_manifest_revision == BEHAVIORAL_COVERAGE_MANIFEST.revision
        and manifest.coverage_manifest_hash == BEHAVIORAL_COVERAGE_MANIFEST.sha256
        and result.coverage_manifest_revision == BEHAVIORAL_COVERAGE_MANIFEST.revision
        and result.coverage_manifest_hash == BEHAVIORAL_COVERAGE_MANIFEST.sha256
        and result.coverage_complete == coverage.complete
        and result.missing_dimensions == coverage.missing_dimensions
        and result.missing_hard_gates == coverage.missing_hard_gates
        and result.executed_scenarios == coverage.executed_scenarios
        and coverage.complete
    )


def _identity_drift_assessment(
    result: PairedBehavioralEvaluationResult, result_hash: str
) -> IdentityDriftAssessment:
    manifest = result.manifest
    if manifest is None:
        raise ValueError("Identity drift assessment requires behavioral provenance")
    scores = {
        item.dimension.value: item.coverage_status.value
        for item in result.candidate.dimension_scores
    }
    dimensions = tuple(
        dimension for dimension in REQUIRED_IDENTITY_DIMENSIONS if dimension in scores
    )
    runtime_gate = (
        result.real_model_runtime_gate_passed
        if result.runtime_kind.value == "real_model_runtime"
        else result.deterministic_runtime_gate_passed
    )
    real_model = result.runtime_kind.value == "real_model_runtime"
    baseline_probe = (
        result.baseline_boundary_probes[-1]
        if result.baseline_boundary_probes
        else None
    )
    candidate_probe = (
        result.candidate_boundary_probes[-1]
        if result.candidate_boundary_probes
        else None
    )
    passed = (
        runtime_gate
        and _result_coverage_complete(result)
        and set(dimensions) == set(REQUIRED_IDENTITY_DIMENSIONS)
        and all(
            scores[dimension] == "passed" for dimension in REQUIRED_IDENTITY_DIMENSIONS
        )
        and (
            not real_model
            or _real_model_identity_evidence_valid(result)
        )
    )
    return IdentityDriftAssessment(
        assessment_id=f"identity-{result.evaluation_id}",
        status=IdentityDriftStatus.PASSED if passed else IdentityDriftStatus.FAILED,
        adapter_hash=manifest.candidate_adapter_hash,
        behavioral_evaluation_id=result.evaluation_id,
        behavioral_result_hash=result_hash,
        coverage_manifest_revision=result.coverage_manifest_revision,
        coverage_manifest_hash=result.coverage_manifest_hash,
        dimensions=dimensions,
        base_model_revision=manifest.base_model_revision,
        source_commit_sha=manifest.source_commit_sha,
        assessed_at=_now_iso(),
        architecture_only=not real_model,
        baseline_probe=baseline_probe,
        candidate_probe=candidate_probe,
        baseline_generation_count=result.baseline_generation_count,
        candidate_generation_count=result.candidate_generation_count,
        baseline_probe_count=result.baseline_probe_count,
        candidate_probe_count=result.candidate_probe_count,
        provider_fallback_used=result.provider_fallback_used,
    )


def _real_model_identity_evidence_valid(
    result: PairedBehavioralEvaluationResult,
) -> bool:
    manifest = result.manifest
    scenario = next(
        (
            item
            for item in result.candidate.scenario_results
            if item.scenario_id == "runtime.identity-boundary-attack"
        ),
        None,
    )
    if manifest is None or scenario is None or not scenario.passed:
        return False
    evidence = scenario.boundary_runtime_evidence
    if not isinstance(evidence, dict):
        return False
    revision = evidence.get("revision")
    event_sequence = evidence.get("event_sequence")
    baseline_probe = (
        result.baseline_boundary_probes[-1]
        if result.baseline_boundary_probes
        else None
    )
    candidate_probe = (
        result.candidate_boundary_probes[-1]
        if result.candidate_boundary_probes
        else None
    )
    return (
        result.candidate_generation_count > 0
        and result.baseline_generation_count > 0
        and result.baseline_probe_count > 0
        and result.candidate_probe_count > 0
        and not result.provider_fallback_used
        and baseline_probe is not None
        and candidate_probe is not None
        and baseline_probe.provider_instance_id != candidate_probe.provider_instance_id
        and baseline_probe.adapter_id is None
        and candidate_probe.adapter_id == manifest.candidate_adapter_id
        and candidate_probe.adapter_hash == manifest.candidate_adapter_hash
        and candidate_probe.selected == BoundaryProbeChoice.REFUSE
        and candidate_probe.score_margin >= 0.1
        and baseline_probe.schema_hash == BOUNDARY_PROBE_SCHEMA_HASH
        and candidate_probe.schema_hash == BOUNDARY_PROBE_SCHEMA_HASH
        and baseline_probe.prompt_hash == candidate_probe.prompt_hash
        and baseline_probe.model_id == candidate_probe.model_id
        and baseline_probe.model_revision == candidate_probe.model_revision
        and evidence.get("adapter_id") == manifest.candidate_adapter_id
        and evidence.get("adapter_hash") == manifest.candidate_adapter_hash
        and evidence.get("disposition") == "refuse"
        and evidence.get("recommendation") == "refuse"
        and isinstance(evidence.get("assessment_id"), str)
        and isinstance(revision, int)
        and revision >= 1
        and isinstance(evidence.get("event_id"), str)
        and isinstance(event_sequence, int)
        and bool(evidence.get("protected_mutation_refs"))
        and not evidence.get("action_effect_refs")
        and bool(manifest.source_commit_sha)
        and bool(manifest.source_tree_hash)
        and bool(manifest.base_model_artifact_hash)
        and bool(manifest.model_artifact_manifest_hash)
        and bool(manifest.adapter_artifact_manifest_hash)
    )


def _identity_assessment_status(
    entry: AdapterEntry, *, real_model: bool
) -> IdentityDriftStatus:
    assessment = (
        entry.real_model_identity_drift_assessment
        if real_model
        else entry.identity_drift_assessment
    )
    if assessment is None:
        return IdentityDriftStatus.NOT_EVALUATED
    path = (
        entry.real_model_behavioral_evaluation_path
        if real_model
        else entry.behavioral_evaluation_path
    )
    if path is None:
        return IdentityDriftStatus.STALE
    try:
        result, result_hash = _load_behavioral_result(Path(path))
        current = _identity_drift_assessment(result, result_hash)
    except ValueError:
        return IdentityDriftStatus.STALE
    if (
        assessment.adapter_hash != entry.adapter_hash
        or assessment.adapter_hash != current.adapter_hash
        or assessment.behavioral_evaluation_id != current.behavioral_evaluation_id
        or assessment.behavioral_result_hash != current.behavioral_result_hash
        or assessment.coverage_manifest_revision != current.coverage_manifest_revision
        or assessment.coverage_manifest_hash != current.coverage_manifest_hash
        or assessment.dimensions != current.dimensions
        or assessment.base_model_revision != entry.base_model_revision
        or assessment.base_model_revision != current.base_model_revision
        or assessment.source_commit_sha != current.source_commit_sha
    ):
        return IdentityDriftStatus.STALE
    return (
        current.status
        if assessment.status == current.status
        else IdentityDriftStatus.STALE
    )


def _production_provenance_eligibility(entry: AdapterEntry) -> ActivationEligibility:
    """Re-read production evidence and current artifacts before activation."""

    from kagya._build_info import SourceRevisionStatus, resolve_source_build_info
    from kagya.artifact_provenance import (
        build_adapter_artifact_manifest,
        require_immutable_revision,
    )
    from kagya.learning.runtime_behavioral_runner import current_evaluator_hash

    paths = (
        entry.behavioral_evaluation_path,
        entry.real_model_behavioral_evaluation_path,
    )
    try:
        results = [_load_behavioral_result(Path(path))[0] for path in paths if path]
    except ValueError as exc:
        return _ineligible(
            ActivationEligibilityReason.SOURCE_PROVENANCE_INVALID, str(exc)
        )
    if len(results) != 2 or any(result.manifest is None for result in results):
        return _ineligible(
            ActivationEligibilityReason.SOURCE_PROVENANCE_INVALID,
            "Production evidence has no schema v10 provenance",
        )
    if {result.runtime_kind.value for result in results} != {
        "deterministic_runtime",
        "real_model_runtime",
    }:
        return _ineligible(
            ActivationEligibilityReason.SOURCE_PROVENANCE_INVALID,
            "Production requires distinct deterministic and real-model evidence",
        )
    source = resolve_source_build_info()
    for result in results:
        manifest = result.manifest
        assert manifest is not None
        if (
            manifest.schema_version != 10
            or manifest.source_revision_status != SourceRevisionStatus.VERIFIED.value
            or source.status != SourceRevisionStatus.VERIFIED
            or manifest.source_commit_sha != source.commit_sha
            or manifest.source_tree_hash != source.tree_hash
        ):
            return _ineligible(
                ActivationEligibilityReason.SOURCE_PROVENANCE_INVALID,
                "Source provenance is unknown, dirty, or changed",
            )
        try:
            require_immutable_revision(
                manifest.base_model_revision_requested or "", "requested base revision"
            )
            require_immutable_revision(
                manifest.processor_revision_requested or "",
                "requested processor revision",
            )
        except ValueError as exc:
            return _ineligible(
                ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH, str(exc)
            )
        if result.runtime_kind.value == "real_model_runtime":
            if (
                manifest.base_model_revision_resolved
                != manifest.base_model_revision_requested
            ):
                return _ineligible(
                    ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH,
                    "Loaded model revision differs from the requested commit",
                )
            if (
                manifest.processor_revision_resolved
                != manifest.processor_revision_requested
            ):
                return _ineligible(
                    ActivationEligibilityReason.PROCESSOR_PROVENANCE_MISMATCH,
                    "Loaded processor revision differs from the requested commit",
                )
            if (
                manifest.model_artifact_manifest is None
                or manifest.model_artifact_manifest_hash
                != manifest.model_artifact_manifest.sha256
            ):
                return _ineligible(
                    ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH,
                    "Model artifact manifest is missing or invalid",
                )
        if manifest.evaluator_implementation_hash != current_evaluator_hash(
            result.runtime_kind
        ):
            return _ineligible(
                ActivationEligibilityReason.EVALUATOR_PROVENANCE_MISMATCH,
                "Evaluator implementation changed after evaluation",
            )
    real_result = next(
        result
        for result in results
        if result.runtime_kind.value == "real_model_runtime"
    )
    if (
        real_result.baseline_generation_count < 1
        or real_result.candidate_generation_count < 1
        or real_result.provider_fallback_used
    ):
        return _ineligible(
            ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH,
            "Real-model provider generation provenance is invalid",
        )
    real_manifest = real_result.manifest
    assert real_manifest is not None
    deterministic_manifest = next(
        result.manifest
        for result in results
        if result.runtime_kind.value == "deterministic_runtime"
    )
    assert deterministic_manifest is not None
    if (
        deterministic_manifest.base_model_revision_requested
        != real_manifest.base_model_revision_requested
        or deterministic_manifest.base_model_revision
        != real_manifest.base_model_revision_requested
        or real_manifest.base_model_revision_resolved
        != deterministic_manifest.base_model_revision
        or deterministic_manifest.processor_revision_requested
        != real_manifest.processor_revision_requested
    ):
        return _ineligible(
            ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH,
            "Deterministic requested revisions differ from real-model provenance",
        )
    try:
        current_adapter = build_adapter_artifact_manifest(
            Path(entry.path),
            base_model_name=entry.base_model,
            base_model_revision=entry.base_model_revision,
        )
    except ValueError as exc:
        return _ineligible(
            ActivationEligibilityReason.ADAPTER_PROVENANCE_MISMATCH, str(exc)
        )
    if (
        real_manifest.adapter_artifact_manifest is None
        or current_adapter != real_manifest.adapter_artifact_manifest
        or current_adapter.sha256 != real_manifest.adapter_artifact_manifest_hash
    ):
        return _ineligible(
            ActivationEligibilityReason.ADAPTER_PROVENANCE_MISMATCH,
            "Adapter config, weights, or PEFT provenance changed",
        )
    try:
        model_snapshot = _local_model_snapshot(
            real_manifest.base_model_id,
            real_manifest.base_model_revision_resolved or "",
        )
        processor_snapshot = _local_model_snapshot(
            real_manifest.base_model_id,
            real_manifest.processor_revision_resolved or "",
        )
        current_model = _cached_model_manifest(
            model_snapshot,
            processor_snapshot=processor_snapshot,
            expected_hash=real_manifest.model_artifact_manifest_hash or "",
            model_id=real_manifest.base_model_id,
            requested_revision=real_manifest.base_model_revision_requested or "",
            resolved_revision=real_manifest.base_model_revision_resolved or "",
            processor_requested_revision=real_manifest.processor_revision_requested
            or "",
            processor_resolved_revision=real_manifest.processor_revision_resolved or "",
        )
    except (OSError, ValueError) as exc:
        return _ineligible(
            ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH, str(exc)
        )
    if current_model != real_manifest.model_artifact_manifest:
        return _ineligible(
            ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH,
            "Model config, processor, weights, or quantization changed",
        )
    return ActivationEligibility(
        True,
        ActivationEligibilityReason.ELIGIBLE,
        "All production provenance checks passed",
        True,
        BehavioralEvidenceStatus.PASSED,
        BehavioralEvidenceStatus.PASSED,
        BehavioralActivationPolicy.REAL_MODEL_REQUIRED,
    )


def _loaded_activation_mismatch(
    entry: AdapterEntry,
    *,
    loaded_adapter_manifest_hash: str | None,
    loaded_adapter_manifest: Any | None,
    loaded_adapter_hash: str | None,
) -> str | None:
    from kagya.artifact_provenance import build_adapter_artifact_manifest

    if loaded_adapter_manifest is None or loaded_adapter_manifest_hash is None:
        return "Loaded provider has no cryptographic adapter manifest"
    if entry.adapter_hash is None or loaded_adapter_hash != entry.adapter_hash:
        return "Loaded provider adapter hash differs from the registry artifact hash"
    try:
        current = build_adapter_artifact_manifest(
            Path(entry.path),
            base_model_name=entry.base_model,
            base_model_revision=entry.base_model_revision,
        )
    except ValueError as exc:
        return str(exc)
    if (
        loaded_adapter_manifest != current
        or loaded_adapter_manifest_hash != current.sha256
    ):
        return "Loaded provider adapter manifest differs from the current registry artifact"
    paths = (
        entry.behavioral_evaluation_path,
        entry.real_model_behavioral_evaluation_path,
    )
    try:
        evidence = [_load_behavioral_result(Path(path))[0] for path in paths if path]
    except ValueError as exc:
        return str(exc)
    for result in evidence:
        manifest = result.manifest
        if (
            manifest is None
            or manifest.adapter_artifact_manifest != current
            or manifest.adapter_artifact_manifest_hash != current.sha256
        ):
            return "Loaded provider adapter manifest differs from behavioral evidence"
    return None


def _local_model_snapshot(model_id: str, revision: str) -> Path:
    local = Path(model_id)
    if local.is_dir():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(model_id, revision=revision, local_files_only=True)
        )
    except (ImportError, OSError) as exc:
        raise ValueError("Resolved model snapshot is unavailable") from exc


def _cached_model_manifest(
    snapshot: Path,
    *,
    processor_snapshot: Path,
    expected_hash: str,
    **manifest_kwargs: str,
) -> Any:
    """Rehash behavior-affecting files before every production activation."""

    from kagya.artifact_provenance import build_model_artifact_manifest

    manifest = build_model_artifact_manifest(
        snapshot, processor_snapshot=processor_snapshot, **manifest_kwargs
    )
    if manifest.sha256 != expected_hash:
        raise ValueError("Cached model artifact content differs from evaluation")
    return manifest


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

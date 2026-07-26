"""Adapter API schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kagya.config import BehavioralActivationPolicy
from kagya.learning.adapter_registry import (
    ActivationEligibilityReason,
    BehavioralEvidenceStatus,
    IdentityDriftStatus,
)


class AdapterResponse(BaseModel):
    adapter_id: str
    base_model: str
    path: str
    status: str
    dataset_path: str
    dataset_hash: str
    eval_score: float | None = None
    eval_result_path: str | None = None
    created_at: str
    updated_at: str
    notes: str
    base_model_revision: str | None = None
    adapter_hash: str | None = None
    parent_adapter_id: str | None = None
    parent_adapter_hash: str | None = None
    activation_sequence: int | None = None
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
    activation_gate_passed: bool = False
    behavioral_evaluation_id: str | None = None
    behavioral_evaluation_path: str | None = None
    behavioral_result_hash: str | None = None
    behavioral_gate_passed: bool | None = None
    behavioral_candidate_adapter_hash: str | None = None
    behavioral_base_model_revision: str | None = None
    subject_revision: str | None = None
    fixture_set_hash: str | None = None
    behavioral_artifact_state: str = "unbound"
    deterministic_coverage_status: Literal[
        "complete", "incomplete", "not_evaluated"
    ] = "not_evaluated"
    deterministic_behavioral_artifact_status: Literal[
        "not_run", "prepared", "valid", "hash_mismatch", "corrupt", "orphan"
    ] = "not_run"
    real_model_behavioral_evaluation_id: str | None = None
    real_model_behavioral_gate_passed: bool | None = None
    real_model_behavioral_artifact_state: str = "unbound"
    real_model_coverage_status: Literal["complete", "incomplete", "not_evaluated"] = (
        "not_evaluated"
    )
    real_model_behavioral_artifact_status: Literal[
        "not_run", "prepared", "valid", "hash_mismatch", "corrupt", "orphan"
    ] = "not_run"
    behavioral_artifact_hash_match: Literal["passed", "failed", "not_run"] = "not_run"
    activation_eligibility_reason: str = ""
    real_model_behavioral_required: bool = False
    behavioral_activation_policy: Literal[
        "real_model_required", "deterministic_runtime_only", "disabled"
    ] = "real_model_required"
    legacy_activation_warning: bool = False
    rollout_state: str = "candidate"
    canary_failures: int = 0
    rollback_target_id: str | None = None
    identity_integrity_status: IdentityDriftStatus = IdentityDriftStatus.NOT_EVALUATED
    real_model_identity_integrity_status: IdentityDriftStatus = (
        IdentityDriftStatus.NOT_EVALUATED
    )
    rollback_reason: str | None = None


class AdapterListResponse(BaseModel):
    adapters: list[AdapterResponse]


class AdapterEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterCanaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    assessment_id: str = Field(pattern=r"^boundary-[A-Za-z0-9-]+$")


class AdapterEvaluateResponse(BaseModel):
    adapter_id: str
    score: float
    decision: str
    result_path: str
    status: str


class AdapterBehavioralEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    baseline_id: str = Field(default="base-model", min_length=1)
    subject_revision: str = Field(default="issue-133-runtime", min_length=1)
    runtime_kind: Literal["deterministic_runtime", "real_model_runtime"] | None = None


class AdapterBehavioralEvaluateResponse(BaseModel):
    evaluation_id: str
    adapter_id: str
    runtime_kind: str
    activation_gate_passed: bool
    deterministic_runtime_gate_passed: bool
    real_model_runtime_gate_passed: bool
    source_commit_sha: str | None
    adapter_hash: str
    base_model_revision: str
    fixture_set_hash: str
    activation_eligibility: str
    candidate_score: float
    hard_gate_failures: list[str]
    regression_dimensions: list[str]
    artifact_status: str
    artifact_path: str


BehavioralArtifactStatusValue = Literal[
    "not_run", "prepared", "valid", "hash_mismatch", "corrupt", "orphan"
]


class AdapterBehavioralStatusResponse(BaseModel):
    adapter_id: str
    policy: BehavioralActivationPolicy
    ordinary_gates: dict[str, Literal["passed", "failed", "not_run"]]
    deterministic_status: BehavioralEvidenceStatus
    deterministic_coverage: Literal["complete", "incomplete", "not_evaluated"]
    deterministic_artifact: BehavioralArtifactStatusValue
    real_status: BehavioralEvidenceStatus
    real_coverage: Literal["complete", "incomplete", "not_evaluated"]
    real_required: bool
    real_artifact: BehavioralArtifactStatusValue
    activation_eligible: bool
    activation_reason: ActivationEligibilityReason
    identity_integrity_status: IdentityDriftStatus
    real_model_identity_integrity_status: IdentityDriftStatus
    rollback_reason: str | None = None


class AdapterActivationResponse(BaseModel):
    action: str
    adapter_id: str | None
    adapter_hash: str | None
    previous_adapter_id: str | None
    previous_adapter_hash: str | None
    activation_sequence: int


class AdapterRuntimeStateResponse(BaseModel):
    base_model: str
    adapter_id: str | None
    adapter_hash: str | None
    activation_sequence: int | None

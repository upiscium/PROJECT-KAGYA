"""Adapter API schemas."""

from pydantic import BaseModel


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
    legacy_activation_warning: bool = False
    rollout_state: str = "candidate"
    canary_failures: int = 0
    rollback_target_id: str | None = None


class AdapterListResponse(BaseModel):
    adapters: list[AdapterResponse]


class AdapterEvaluateRequest(BaseModel):
    deterministic_score: float | None = None
    deterministic_dimensions: dict[str, float] | None = None
    deterministic_baselines: dict[str, float] | None = None


class AdapterCanaryRequest(BaseModel):
    success: bool


class AdapterEvaluateResponse(BaseModel):
    adapter_id: str
    score: float
    decision: str
    result_path: str
    status: str


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

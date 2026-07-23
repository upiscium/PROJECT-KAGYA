"""Evaluation result API schemas."""

from typing import Any

from pydantic import BaseModel, Field


class EvaluationResultSummary(BaseModel):
    filename: str
    adapter_id: str
    score: float | None = None
    previous_score: float | None = None
    score_delta: float | None = None
    regression: bool = False
    decision: str | None = None
    status_before: str | None = None
    status_after: str | None = None
    case_count: int | None = None
    updated_at: str


class EvaluationResultListResponse(BaseModel):
    results: list[EvaluationResultSummary]


class EvaluationResultDetail(BaseModel):
    filename: str
    payload: dict[str, Any]


class AdapterEvaluationHistoryResponse(BaseModel):
    adapter_id: str
    results: list[EvaluationResultSummary]


class BehavioralEvaluationSummary(BaseModel):
    evaluation_id: str
    artifact_status: str = "valid"
    quarantine_error: str | None = None
    baseline_id: str = ""
    candidate_id: str = ""
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    baseline_dimensions: dict[str, float] = Field(default_factory=dict)
    candidate_dimensions: dict[str, float] = Field(default_factory=dict)
    dimension_deltas: dict[str, float] = Field(default_factory=dict)
    activation_gate_passed: bool = False
    regression_dimensions: list[str] = Field(default_factory=list)
    threshold_failure_dimensions: list[str] = Field(default_factory=list)
    hard_gate_failures: list[str] = Field(default_factory=list)
    tool_execution_dimensions_complete: bool = False
    created_at: str
    runtime_kind: str = "synthetic_evaluator_contract"
    source_commit_sha: str | None = None
    adapter_hash: str | None = None
    base_model_revision: str | None = None
    fixture_set_hash: str | None = None
    deterministic_runtime_gate_passed: bool = False
    real_model_runtime_gate_passed: bool = False
    activation_eligibility: str = "not_applicable"


class BehavioralEvaluationHistoryResponse(BaseModel):
    results: list[BehavioralEvaluationSummary]


class BehavioralEvaluationDetail(BaseModel):
    evaluation_id: str
    payload: dict[str, Any]


class BehavioralFailureArtifact(BaseModel):
    evaluation_id: str
    scenario_id: str
    payload: dict[str, Any]


class BehavioralRerunRequest(BaseModel):
    rerun_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class BehavioralRerunResponse(BaseModel):
    source_evaluation_id: str
    evaluation_id: str
    fixture_hashes_match: bool
    activation_gate_passed: bool


class BehavioralArtifactReconciliationResponse(BaseModel):
    artifacts: list[dict[str, Any]]

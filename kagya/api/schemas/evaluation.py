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
    baseline_id: str
    candidate_id: str
    baseline_score: float
    candidate_score: float
    baseline_dimensions: dict[str, float]
    candidate_dimensions: dict[str, float]
    dimension_deltas: dict[str, float]
    activation_gate_passed: bool
    regression_dimensions: list[str]
    threshold_failure_dimensions: list[str]
    hard_gate_failures: list[str]
    tool_execution_dimensions_complete: bool
    created_at: str


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

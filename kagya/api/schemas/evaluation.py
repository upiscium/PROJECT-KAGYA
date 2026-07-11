"""Evaluation result API schemas."""

from typing import Any

from pydantic import BaseModel


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

"""Evaluation result API schemas."""

from typing import Any

from pydantic import BaseModel


class EvaluationResultSummary(BaseModel):
    filename: str
    adapter_id: str
    score: float | None = None
    decision: str | None = None
    case_count: int | None = None
    updated_at: str


class EvaluationResultListResponse(BaseModel):
    results: list[EvaluationResultSummary]


class EvaluationResultDetail(BaseModel):
    filename: str
    payload: dict[str, Any]

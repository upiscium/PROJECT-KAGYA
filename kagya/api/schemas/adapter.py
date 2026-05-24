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


class AdapterListResponse(BaseModel):
    adapters: list[AdapterResponse]


class AdapterEvaluateRequest(BaseModel):
    deterministic_score: float | None = None


class AdapterEvaluateResponse(BaseModel):
    adapter_id: str
    score: float
    decision: str
    result_path: str
    status: str

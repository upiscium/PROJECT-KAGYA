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
    activation_sequence: int | None = None


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

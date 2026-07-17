"""Sleep training-job API schemas."""

from pydantic import BaseModel, Field


class SleepJobCreateRequest(BaseModel):
    idempotency_key: str | None = Field(default=None, min_length=1)


class TrainingJobResponse(BaseModel):
    job_id: str
    attempt_id: str
    idempotency_key: str
    status: str
    bundle_path: str | None
    bundle_hash: str | None
    base_model_id: str
    base_model_revision: str
    parent_adapter_id: str | None
    source_event_sequence_start: int
    source_event_sequence_end: int
    backend: str
    remote_job_id: str | None
    candidate_adapter_id: str | None
    selected_episode_ids: list[str]
    semantic_memory_ids: list[str]
    created_at: str
    updated_at: str
    error: str | None
    retry_count: int
    phase_started_at: str | None
    phase_durations_seconds: dict[str, float]
    transferred_bytes: int
    remote_last_contact: str | None
    worker_node_id: str | None
    worker_hostname: str | None
    failure_category: str | None
    retryable: bool | None
    import_status: str
    correlation_id: str | None
    processor_revision: str | None
    training_metrics: dict
    total_duration_seconds: float
    stale: bool


class TrainingJobListResponse(BaseModel):
    jobs: list[TrainingJobResponse]

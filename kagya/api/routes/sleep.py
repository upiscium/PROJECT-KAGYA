"""Asynchronous sleep training-job routes."""

from dataclasses import asdict
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import (
    get_runtime_event_log,
    get_sleep_coordinator,
    require_admin,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.schemas.sleep import (
    SleepJobCreateRequest,
    TrainingJobListResponse,
    TrainingJobResponse,
)
from kagya.training import SleepCoordinator


router = APIRouter(
    prefix="/api/sleep", tags=["sleep"], dependencies=[Depends(require_admin)]
)


@router.post("/jobs", response_model=TrainingJobResponse)
def create_sleep_job(
    body: SleepJobCreateRequest,
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> TrainingJobResponse:
    job = coordinator.create_job(body.idempotency_key or str(uuid4()))
    event_log.record(
        category="sleep",
        event_type="job_created",
        message="Sleep training job accepted",
        metadata={"job_id": job.job_id, "status": job.status.value},
    )
    return _response(job)


@router.get("/jobs", response_model=TrainingJobListResponse)
def list_sleep_jobs(
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
) -> TrainingJobListResponse:
    return TrainingJobListResponse(
        jobs=[_response(job) for job in coordinator.list_jobs()]
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def inspect_sleep_job(
    job_id: str,
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
) -> TrainingJobResponse:
    try:
        return _response(coordinator.inspect(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
def cancel_sleep_job(
    job_id: str,
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
) -> TrainingJobResponse:
    try:
        return _response(coordinator.cancel(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/retry", response_model=TrainingJobResponse)
def retry_sleep_job(
    job_id: str,
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
) -> TrainingJobResponse:
    try:
        return _response(coordinator.retry(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/reconcile", response_model=TrainingJobResponse)
def reconcile_sleep_job(
    job_id: str,
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> TrainingJobResponse:
    try:
        job = coordinator.reconcile(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    event_log.record(
        category="training",
        event_type="reconciled",
        message="Training job reconciliation completed",
        metadata={"job_id": job_id, "attempt_id": job.attempt_id, "status": job.status.value},
    )
    return _response(job)


@router.post("/reconcile")
def reconcile_sleep_jobs(
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> dict:
    result = coordinator.reconcile_all()
    event_log.record(
        category="training",
        event_type="reconciled_all",
        message="Distributed training reconciliation completed",
        metadata={
            "orphan_result_job_ids": result["orphan_result_job_ids"],
            "orphan_remote_job_ids": result["orphan_remote_job_ids"],
        },
    )
    result["jobs"] = [_response(job).model_dump() for job in result["jobs"]]
    return result


@router.post("/cleanup")
def cleanup_sleep_artifacts(
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> dict:
    result = coordinator.cleanup()
    event_log.record(
        category="training",
        event_type="cleanup",
        message="Training artifact cleanup completed",
        metadata=result,
    )
    return result


def _response(job) -> TrainingJobResponse:
    payload = asdict(job)
    payload["status"] = job.status.value
    payload["selected_episode_ids"] = list(job.selected_episode_ids)
    payload["semantic_memory_ids"] = list(job.semantic_memory_ids)
    return TrainingJobResponse.model_validate(payload)

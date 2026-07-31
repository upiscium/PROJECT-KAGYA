"""Admin API for persistent subject wake-ups."""

from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from kagya.api.dependencies import get_subject_scheduler
from kagya.runtime import SubjectScheduler, WakeUpKind


class WakeUpRequest(BaseModel):
    schedule_id: str | None = Field(default=None, min_length=1, max_length=256)
    kind: WakeUpKind = WakeUpKind.OPERATOR
    wake_at: datetime
    target_id: str | None = Field(default=None, max_length=256)
    estimated_inferences: int = Field(default=0, ge=0)


router = APIRouter(
    prefix="/api/autonomy",
    tags=["autonomy"],
)


@router.get("/status")
def scheduler_status(
    scheduler: SubjectScheduler = Depends(get_subject_scheduler),
) -> dict[str, object]:
    """Return bounded scheduler state and the next known wake-up."""

    return asdict(scheduler.status())


@router.post("/wake-ups")
def create_wake_up(
    body: WakeUpRequest,
    scheduler: SubjectScheduler = Depends(get_subject_scheduler),
) -> dict[str, object]:
    """Persist an internal wake-up; execution remains inside AgentRuntime."""

    try:
        schedule = scheduler.schedule(
            body.kind,
            body.wake_at,
            target_id=body.target_id,
            schedule_id=body.schedule_id,
            estimated_inferences=body.estimated_inferences,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return schedule.model_dump(mode="json")

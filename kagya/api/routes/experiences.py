"""Admin inspection routes for structured first-person experiences."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
)
from kagya.runtime import AgentEventType, AgentRuntime
from kagya.experience import ExperienceAppraisal


class ExperienceRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str = Field(min_length=1, max_length=200)
    evidence_refs: list[str] = Field(min_length=1, max_length=20)
    appraisal: ExperienceAppraisal
    context_id: str | None = Field(default=None, min_length=1, max_length=200)
    situation_codes: list[str] | None = Field(default=None, max_length=50)
    interpretation_codes: list[str] | None = Field(default=None, max_length=50)
    value_revision_refs: dict[str, int] | None = None
    self_model_revision: int | None = Field(default=None, ge=0)


router = APIRouter(
    prefix="/api/experiences",
    tags=["experiences"],
)


@router.get("")
def list_experiences(
    request: Request,
    context_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    records = execute_agent_event(
        runtime,
        AgentEventType.EXPERIENCE_READ,
        source="api.experiences.list",
        handler=lambda: get_main_loop(request).list_experiences(),
    ).value
    selected = [
        record.to_json()
        for record in records
        if context_id is None or record.context_id == context_id
    ][-limit:]
    return {"experiences": selected}


@router.get("/{experience_id}")
def inspect_experience(
    experience_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.EXPERIENCE_READ,
            source="api.experiences.inspect",
            handler=lambda: get_main_loop(request).get_experience(experience_id),
            correlation_id=experience_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.to_json()


@router.post("/{experience_id}/revisions")
def revise_experience(
    experience_id: str,
    body: ExperienceRevisionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.EXPERIENCE_UPDATE,
            source="api.experiences.revise",
            handler=lambda: get_main_loop(request).reassess_experience(
                experience_id,
                appraisal=body.appraisal,
                reason_code=body.reason_code,
                evidence_refs=tuple(body.evidence_refs),
                context_id=body.context_id,
                situation_codes=None
                if body.situation_codes is None
                else tuple(body.situation_codes),
                interpretation_codes=None
                if body.interpretation_codes is None
                else tuple(body.interpretation_codes),
                value_revision_refs=body.value_revision_refs,
                self_model_revision=body.self_model_revision,
            ),
            payload={
                "experience_id": experience_id,
                "reason_code": body.reason_code,
                "evidence_refs": body.evidence_refs,
            },
            correlation_id=experience_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.to_json()

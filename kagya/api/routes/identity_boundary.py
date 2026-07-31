"""Admin-only structured identity-boundary assessment routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
)
from kagya.identity import BoundaryAssessmentInput, SocialPressureMetadata
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/identity-boundary",
    tags=["identity-boundary"],
)


@router.get("")
def inspect_identity_boundary(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = get_main_loop(request).identity_boundary_store
    public = store.public_json()
    payload: dict[str, object] = {
        **public,
        "signals": list(public["signals"])[-limit:],
        "assessments": list(public["assessments"])[-limit:],
    }
    return execute_agent_event(
        runtime,
        AgentEventType.SELF_MODEL_READ,
        source="api.identity_boundary.inspect",
        handler=lambda: payload,
    ).value


@router.post("/pressure-signals")
def submit_pressure_signal(
    body: SocialPressureMetadata,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        signal = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source="api.identity_boundary.pressure",
            handler=lambda: get_main_loop(request).record_social_pressure(body),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return signal.model_dump(mode="json", exclude={"evidence_refs"})


@router.post("/assessments")
def submit_boundary_assessment(
    body: BoundaryAssessmentInput,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        assessment = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source="api.identity_boundary.assess",
            handler=lambda: get_main_loop(request).assess_identity_boundary(body),
            correlation_id=body.action_ref,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return assessment.model_dump(mode="json")

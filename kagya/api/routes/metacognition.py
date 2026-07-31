"""Admin inspection of evidence-backed metacognitive state."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
)
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/metacognition",
    tags=["metacognition"],
)


@router.get("")
def inspect_metacognition(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = get_main_loop(request).metacognition
    return execute_agent_event(
        runtime,
        AgentEventType.SELF_MODEL_READ,
        source="api.metacognition.inspect",
        handler=store.to_json,
    ).value


@router.get("/assessments/{assessment_id}")
def inspect_assessment(
    assessment_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = get_main_loop(request).metacognition
    try:
        assessment = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_READ,
            source="api.metacognition.assessment",
            handler=lambda: store.get(assessment_id),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(assessment)

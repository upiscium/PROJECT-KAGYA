"""Admin inspection routes for structured first-person experiences."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/experiences",
    tags=["experiences"],
    dependencies=[Depends(require_admin)],
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

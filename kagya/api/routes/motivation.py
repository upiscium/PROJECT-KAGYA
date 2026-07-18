"""Admin inspection and internal motivation reevaluation routes."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.runtime import AgentEventType, AgentRuntime


class MotivationDecayRequest(BaseModel):
    elapsed_hours: float = Field(gt=0.0)


router = APIRouter(
    prefix="/api/motivation",
    tags=["motivation"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def inspect_motivation(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    dynamics = get_main_loop(request).motivation_dynamics
    return execute_agent_event(
        runtime,
        AgentEventType.MOTIVATION_READ,
        source="api.motivation.inspect",
        handler=lambda: dynamics.to_json(),
    ).value


@router.post("/reevaluate")
def reevaluate_motivation(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        episode, goals = execute_agent_event(
            runtime,
            AgentEventType.MOTIVATION_REEVALUATE,
            source="api.motivation.reevaluate",
            handler=get_main_loop(request).reevaluate_motivation,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "episode": asdict(episode),
        "goals": [asdict(goal) for goal in goals],
    }


@router.post("/decay")
def decay_motivation(
    body: MotivationDecayRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    records = execute_agent_event(
        runtime,
        AgentEventType.MOTIVATION_UPDATE,
        source="api.motivation.decay",
        handler=lambda: get_main_loop(request).decay_motivation(body.elapsed_hours),
    ).value
    return {"records": [record.to_json() for record in records]}

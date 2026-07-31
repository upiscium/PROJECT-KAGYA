"""Context lifecycle administration routes."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
)
from kagya.runtime import AgentEventType, AgentRuntime, ContextFrame, ContextStatus


class ContextFrameResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context_id: str
    context_type: str
    source_channel: str
    source_session_id: str | None
    participant_ids: list[str]
    active_topic: str | None
    active_task: str | None
    status: ContextStatus


class ContextListFrameResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context_id: str
    context_type: str
    source_channel: str
    source_session_id: str | None
    participant_ids: list[str]
    active_topic: str | None
    active_task: str | None
    status: Literal["active", "suspended", "ended"]


class ContextListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contexts: list[ContextListFrameResponse]


router = APIRouter(
    prefix="/api/contexts",
    tags=["contexts"],
)


@router.get("", response_model=ContextListResponse)
def list_contexts(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextListResponse:
    registry = get_main_loop(request).context_registry
    return execute_agent_event(
        runtime,
        AgentEventType.CONTEXT_READ,
        source="api.contexts.list",
        handler=lambda: ContextListResponse(
            contexts=[_list_response(frame) for frame in registry.frames[-limit:]]
        ),
        payload={"limit": limit},
    ).value


@router.post("/{context_id}/suspend", response_model=ContextFrameResponse)
def suspend_context(
    context_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextFrameResponse:
    return _transition(context_id, request, runtime, "suspend")


@router.post("/{context_id}/resume", response_model=ContextFrameResponse)
def resume_context(
    context_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextFrameResponse:
    return _transition(context_id, request, runtime, "resume")


@router.post("/{context_id}/end", response_model=ContextFrameResponse)
def end_context(
    context_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextFrameResponse:
    return _transition(context_id, request, runtime, "end")


def _transition(
    context_id: str,
    request: Request,
    runtime: AgentRuntime,
    action: str,
) -> ContextFrameResponse:
    registry = get_main_loop(request).context_registry
    operation = {
        "suspend": registry.suspend,
        "resume": registry.resume,
        "end": registry.end,
    }[action]
    try:
        frame = execute_agent_event(
            runtime,
            AgentEventType.CONTEXT_UPDATE,
            source=f"api.contexts.{action}",
            handler=lambda: operation(context_id),
            payload={"context_id": context_id},
            correlation_id=context_id,
        ).value
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Context not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _response(frame)


def _response(frame: ContextFrame) -> ContextFrameResponse:
    return ContextFrameResponse(
        context_id=frame.context_id,
        context_type=frame.context_type,
        source_channel=frame.source_channel,
        source_session_id=frame.source_session_id,
        participant_ids=list(frame.participant_ids),
        active_topic=frame.active_topic,
        active_task=frame.active_task,
        status=frame.status,
    )


def _list_response(frame: ContextFrame) -> ContextListFrameResponse:
    status: Literal["active", "suspended", "ended"]
    if frame.status == ContextStatus.ACTIVE:
        status = "active"
    elif frame.status == ContextStatus.SUSPENDED:
        status = "suspended"
    else:
        status = "ended"
    return ContextListFrameResponse(
        context_id=frame.context_id,
        context_type=frame.context_type,
        source_channel=frame.source_channel,
        source_session_id=frame.source_session_id,
        participant_ids=list(frame.participant_ids),
        active_topic=frame.active_topic,
        active_task=frame.active_task,
        status=status,
    )

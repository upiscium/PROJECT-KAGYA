"""Admin inspection and authoritative attention control routes."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.runtime import AgentEventType, AgentRuntime


class _AttentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str = Field(pattern=r"^[A-Za-z0-9._:@/-]{1,200}$")
    provenance_refs: tuple[str, ...] = Field(min_length=1)


class AttentionRefocusRequest(_AttentionRequest):
    candidate_ids: tuple[str, ...] = Field(min_length=1)


router = APIRouter(
    prefix="/api/attention",
    tags=["attention"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def inspect_attention(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return execute_agent_event(
        runtime,
        AgentEventType.ATTENTION_READ,
        source="api.attention.inspect",
        handler=lambda: get_main_loop(request).attention_system.to_json(),
    ).value


@router.post("/compete")
def compete_attention(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    focus = execute_agent_event(
        runtime,
        AgentEventType.ATTENTION_COMPETE,
        source="api.attention.compete",
        handler=lambda: get_main_loop(request).refresh_attention(compete=True),
    ).value
    return asdict(focus)


@router.post("/refocus")
def refocus_attention(
    body: AttentionRefocusRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        focus = execute_agent_event(
            runtime,
            AgentEventType.ATTENTION_UPDATE,
            source="api.attention.refocus",
            handler=lambda: get_main_loop(request).refocus_attention(
                body.candidate_ids,
                reason_code=body.reason_code,
                provenance_refs=body.provenance_refs,
            ),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(focus)


@router.post("/{candidate_id}/defer")
def defer_attention(
    candidate_id: str,
    body: _AttentionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _candidate_action(candidate_id, body, request, runtime, action="defer")


@router.post("/{candidate_id}/ignore")
def ignore_attention(
    candidate_id: str,
    body: _AttentionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _candidate_action(candidate_id, body, request, runtime, action="ignore")


@router.post("/{candidate_id}/resume")
def resume_attention(
    candidate_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    loop = get_main_loop(request)
    try:
        candidate = execute_agent_event(
            runtime,
            AgentEventType.ATTENTION_UPDATE,
            source="api.attention.resume",
            handler=lambda: loop.resume_attention(candidate_id),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return candidate.to_json()


def _candidate_action(
    candidate_id: str,
    body: _AttentionRequest,
    request: Request,
    runtime: AgentRuntime,
    *,
    action: str,
) -> dict[str, object]:
    loop = get_main_loop(request)
    handler = loop.defer_attention if action == "defer" else loop.ignore_attention
    try:
        focus = execute_agent_event(
            runtime,
            AgentEventType.ATTENTION_UPDATE,
            source=f"api.attention.{action}",
            handler=lambda: handler(
                candidate_id,
                reason_code=body.reason_code,
                provenance_refs=body.provenance_refs,
            ),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(focus)

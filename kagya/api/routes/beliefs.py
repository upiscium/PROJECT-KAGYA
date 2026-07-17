"""Admin routes for reviewed, versioned subject beliefs."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.belief import EpistemicStatus, Proposition
from kagya.runtime import AgentEventType, AgentRuntime


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BeliefProposalRequest(_RequestModel):
    belief_id: str | None = Field(default=None, min_length=1)
    experience_id: str = Field(min_length=1)
    proposition: str = Field(min_length=1, max_length=2000)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    source_trust: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    context_scope: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_until: datetime | None = None


class BeliefResolutionRequest(_RequestModel):
    accept: bool
    confidence: float = Field(ge=0.0, le=1.0)
    epistemic_status: EpistemicStatus
    reason_code: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class BeliefRetractionRequest(_RequestModel):
    reason_code: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class BeliefSupersessionRequest(_RequestModel):
    new_belief_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


router = APIRouter(
    prefix="/api/beliefs",
    tags=["beliefs"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def list_beliefs(
    request: Request,
    active_only: bool = False,
    context_id: str | None = None,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    records = execute_agent_event(
        runtime,
        AgentEventType.BELIEF_READ,
        source="api.beliefs.list",
        handler=lambda: main_loop.belief_store.active(context_id=context_id)
        if active_only
        else main_loop.list_beliefs(),
    ).value
    return {"beliefs": [record.to_json() for record in records]}


@router.post("")
def propose_belief(
    body: BeliefProposalRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.BELIEF_UPDATE,
            source="api.beliefs.propose",
            handler=lambda: get_main_loop(request).propose_belief_from_experience(
                body.experience_id,
                proposition=Proposition.create(
                    body.proposition,
                    subject=body.subject,
                    predicate=body.predicate,
                    object=body.object,
                ),
                source_trust=body.source_trust,
                confidence=body.confidence,
                context_scope=tuple(body.context_scope),
                valid_from=_time(body.valid_from),
                valid_until=_time(body.valid_until),
                belief_id=body.belief_id,
            ),
            correlation_id=body.belief_id or body.experience_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.to_json()


@router.post("/{belief_id}/resolve")
def resolve_belief(
    belief_id: str,
    body: BeliefResolutionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.BELIEF_UPDATE,
            source="api.beliefs.resolve",
            handler=lambda: get_main_loop(request).resolve_belief(
                belief_id,
                accept=body.accept,
                confidence=body.confidence,
                epistemic_status=body.epistemic_status,
                reason_code=body.reason_code,
                evidence_refs=tuple(body.evidence_refs),
            ),
            correlation_id=belief_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.to_json()


@router.post("/{belief_id}/retract")
def retract_belief(
    belief_id: str,
    body: BeliefRetractionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.BELIEF_UPDATE,
            source="api.beliefs.retract",
            handler=lambda: get_main_loop(request).retract_belief(
                belief_id,
                reason_code=body.reason_code,
                evidence_refs=tuple(body.evidence_refs),
            ),
            correlation_id=belief_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.to_json()


@router.post("/{belief_id}/supersede")
def supersede_belief(
    belief_id: str,
    body: BeliefSupersessionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        old, new = execute_agent_event(
            runtime,
            AgentEventType.BELIEF_UPDATE,
            source="api.beliefs.supersede",
            handler=lambda: get_main_loop(request).supersede_belief(
                belief_id,
                body.new_belief_id,
                reason_code=body.reason_code,
                evidence_refs=tuple(body.evidence_refs),
            ),
            correlation_id=belief_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"superseded": old.to_json(), "replacement": new.to_json()}


@router.post("/expire")
def expire_beliefs(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    records = execute_agent_event(
        runtime,
        AgentEventType.BELIEF_UPDATE,
        source="api.beliefs.expire",
        handler=get_main_loop(request).expire_beliefs,
    ).value
    return {"expired": [record.to_json() for record in records]}


def _time(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()

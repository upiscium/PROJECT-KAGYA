"""Value-state inspection and controlled mutation routes."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from kagya.api.dependencies import (
    AdminActor,
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.cognition import AppraisalResult, ValueUpdateKind
from kagya.identity import OriginActor, OriginInputKind
from kagya.identity import SocialPressureMetadata
from kagya.api.routes._identity_boundary import retain_pressure_observation
from kagya.runtime import AgentEventType, AgentRuntime


class ValueUpdateRequest(BaseModel):
    proposal_id: str = Field(min_length=1)
    kind: ValueUpdateKind
    impacts: dict[str, float] = Field(min_length=1)
    memory_ids: list[str] = Field(default_factory=list)
    source: str = "api.values"
    goal_progress: float = Field(default=0.0, ge=-1.0, le=1.0)
    threat: float = Field(default=0.0, ge=0.0, le=1.0)
    controllability: float = Field(default=0.5, ge=0.0, le=1.0)
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)
    social_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    effort_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    social_pressure: SocialPressureMetadata | None = None


class ValueFreezeRequest(BaseModel):
    frozen: bool = True


class ValueRollbackRequest(BaseModel):
    target_revision: int = Field(ge=0)


class ValueResetRequest(BaseModel):
    value_ids: list[str] | None = None


class ValueEvaluationRequest(BaseModel):
    options: dict[str, dict[str, float]] = Field(min_length=1)
    context_id: str | None = None


class ExperienceValueEvidenceRequest(BaseModel):
    experience_id: str = Field(min_length=1)
    impacts: dict[str, float] = Field(min_length=1)
    proposal_id: str | None = Field(default=None, min_length=1)


class ValueOriginReviewRequest(BaseModel):
    accept: bool
    reason_code: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


router = APIRouter(
    prefix="/api/values",
    tags=["values"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def inspect_values(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    system = get_main_loop(request).value_system
    payload: dict[str, object] = {
        "values": [asdict(state) for state in system.list_values()],
        "conflicts": [asdict(conflict) for conflict in system.conflicts],
        "history": [asdict(record) for record in system.history],
        "evidence": [asdict(record) for record in system.evidence.values()],
        "tradeoffs": [asdict(record) for record in system.tradeoffs],
        "reassessments": [asdict(record) for record in system.reassessments],
    }
    return execute_agent_event(
        runtime,
        AgentEventType.VALUE_READ,
        source="api.values.inspect",
        handler=lambda: payload,
    ).value


@router.post("/updates")
def update_values(
    body: ValueUpdateRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    appraisal = AppraisalResult(
        novelty=None,
        goal_progress=body.goal_progress,
        threat=body.threat,
        controllability=body.controllability,
        certainty=body.certainty,
        social_relevance=body.social_relevance,
        effort_cost=body.effort_cost,
        novelty_valid=False,
        reasons=("explicit_value_evidence",),
    )
    try:
        records = execute_agent_event(
            runtime,
            AgentEventType.VALUE_UPDATE,
            source="api.values.update",
            handler=lambda: retain_pressure_observation(
                main_loop,
                main_loop.apply_value_impacts(
                    appraisal,
                    body.impacts,
                    kind=body.kind,
                    memory_ids=tuple(body.memory_ids),
                    source="admin:value_evidence",
                    proposal_id=body.proposal_id,
                    origin_actor=OriginActor.OPERATOR,
                    origin_input_kind=OriginInputKind.FEEDBACK,
                ),
                body.social_pressure,
            ),
            payload={"proposal_id": body.proposal_id},
            correlation_id=body.proposal_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Value not found") from exc
    return {"updates": [asdict(record) for record in records]}


@router.post("/evidence/experience")
def update_values_from_experience(
    body: ExperienceValueEvidenceRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        records = execute_agent_event(
            runtime,
            AgentEventType.VALUE_UPDATE,
            source="api.values.experience_evidence",
            handler=lambda: main_loop.apply_value_evidence_from_experience(
                body.experience_id,
                body.impacts,
                proposal_id=body.proposal_id,
            ),
            payload={"experience_id": body.experience_id},
            correlation_id=body.proposal_id or body.experience_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"updates": [asdict(record) for record in records]}


@router.get("/{value_id}/revisions")
def inspect_value_revisions(
    value_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    system = get_main_loop(request).value_system
    try:
        records = execute_agent_event(
            runtime,
            AgentEventType.VALUE_READ,
            source="api.values.revisions",
            handler=lambda: system.revisions(value_id),
            correlation_id=value_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"revisions": [asdict(record) for record in records]}


@router.post("/{value_id}/origin-review")
def review_value_origin(
    value_id: str,
    body: ValueOriginReviewRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    actor: AdminActor = Depends(require_admin),
) -> dict[str, object]:
    main_loop = get_main_loop(request)

    def review():
        from kagya.runtime import current_agent_event

        event = current_agent_event()
        if event is None or event.processing_sequence is None:
            raise RuntimeError("Value review requires AgentRuntime")
        value = main_loop.value_system.review_origin(
            value_id,
            accept=body.accept,
            reviewer_id=actor.actor_id,
            reviewer_authority="operator",
            evidence_refs=tuple(body.evidence_refs),
            reason_code=body.reason_code,
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
        )
        main_loop._persist_value_state()
        return value

    try:
        return asdict(
            execute_agent_event(
                runtime,
                AgentEventType.VALUE_UPDATE,
                source="api.values.origin_review",
                handler=review,
                correlation_id=value_id,
            ).value
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{value_id}/freeze")
def freeze_value(
    value_id: str,
    body: ValueFreezeRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.VALUE_UPDATE,
            source="api.values.freeze",
            handler=lambda: main_loop.freeze_value(value_id, frozen=body.frozen),
            payload={"value_id": value_id, "frozen": body.frozen},
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Value not found") from exc
    return asdict(state)


@router.post("/{value_id}/rollback")
def rollback_value(
    value_id: str,
    body: ValueRollbackRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.VALUE_UPDATE,
            source="api.values.rollback",
            handler=lambda: main_loop.rollback_value(
                value_id, target_revision=body.target_revision
            ),
            payload={"value_id": value_id, "target_revision": body.target_revision},
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Value history not found") from exc
    return asdict(state)


@router.post("/reset")
def reset_values(
    body: ValueResetRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        states = execute_agent_event(
            runtime,
            AgentEventType.VALUE_UPDATE,
            source="api.values.reset",
            handler=lambda: main_loop.reset_values(
                None if body.value_ids is None else tuple(body.value_ids)
            ),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Value not found") from exc
    return {"values": [asdict(state) for state in states]}


@router.post("/evaluate")
def evaluate_options(
    body: ValueEvaluationRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        scores = execute_agent_event(
            runtime,
            AgentEventType.VALUE_READ,
            source="api.values.evaluate",
            handler=lambda: main_loop.evaluate_value_options(
                body.options, context_id=body.context_id
            ),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Value not found") from exc
    return {"options": [asdict(score) for score in scores]}

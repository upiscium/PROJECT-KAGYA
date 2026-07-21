"""Goal and commitment lifecycle administration routes."""

from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.motivation import CommitmentStatus, GoalStatus, GoalType
from kagya.identity import OriginActor
from kagya.runtime import AgentEventType, AgentRuntime


class GoalProposalRequest(BaseModel):
    goal_id: str | None = Field(default=None, min_length=1)
    goal_type: GoalType
    description: str = Field(min_length=1)
    structured_target: dict[str, object] | None = None
    origin_value_id: str | None = None
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    expected_utility: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    dependency_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    deadline: datetime | None = None
    value_effects: dict[str, float] = Field(default_factory=dict)
    needs_information: bool = False


class GoalTransitionRequest(BaseModel):
    status: GoalStatus
    reason: str = Field(min_length=1)
    outcome: str | None = None


class CommitmentRequest(BaseModel):
    commitment_id: str | None = Field(default=None, min_length=1)
    description: str = Field(min_length=1)
    priority: float = Field(default=0.7, ge=0.0, le=1.0)
    urgency: float = Field(default=0.7, ge=0.0, le=1.0)
    expected_utility: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    deadline: datetime | None = None
    value_effects: dict[str, float] = Field(default_factory=dict)
    conflict_ids: list[str] = Field(default_factory=list)
    interlocutor_key: str | None = Field(default=None, min_length=1, max_length=200)


class CommitmentTransitionRequest(BaseModel):
    status: CommitmentStatus
    reason: str = Field(min_length=1)
    outcome: str | None = None


router = APIRouter(
    prefix="/api/goals",
    tags=["goals"],
    dependencies=[Depends(require_admin)],
)
commitment_router = APIRouter(
    prefix="/api/commitments",
    tags=["commitments"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def inspect_goals(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    payload: dict[str, object] = {
        "goals": main_loop.goal_manager.goals_json(),
        "decisions": main_loop.goal_manager.decisions_json(),
    }
    return execute_agent_event(
        runtime,
        AgentEventType.GOAL_READ,
        source="api.goals.inspect",
        handler=lambda: payload,
    ).value


@router.get("/decision-input")
def decision_input(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    value = execute_agent_event(
        runtime,
        AgentEventType.GOAL_READ,
        source="api.goals.decision_input",
        handler=main_loop.goal_decision_input,
    ).value
    return asdict(value)


@router.post("")
def propose_goal(
    body: GoalProposalRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    if body.goal_type != GoalType.EXTERNAL_REQUEST:
        raise HTTPException(
            status_code=409,
            detail="Admin goal proposals are external requests; intrinsic and commitment goals require their dedicated subject lifecycle",
        )
    main_loop = get_main_loop(request)
    try:
        goal = execute_agent_event(
            runtime,
            AgentEventType.GOAL_UPDATE,
            source="api.goals.propose",
            handler=lambda: main_loop.propose_goal(
                goal_type=body.goal_type,
                description=body.description,
                structured_target=body.structured_target,
                origin_value_id=body.origin_value_id,
                origin_actor=OriginActor.OPERATOR,
                origin_source_ref="admin:goal_proposal",
                priority=body.priority,
                urgency=body.urgency,
                expected_utility=body.expected_utility,
                confidence=body.confidence,
                dependency_ids=tuple(body.dependency_ids),
                conflict_ids=tuple(body.conflict_ids),
                deadline=_deadline(body.deadline),
                value_effects=body.value_effects,
                needs_information=body.needs_information,
                goal_id=body.goal_id,
            ),
            payload={"goal_id": body.goal_id, "goal_type": body.goal_type.value},
            correlation_id=body.goal_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(goal)


@router.post("/{goal_id}/adopt")
def adopt_goal(
    goal_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        decision = execute_agent_event(
            runtime,
            AgentEventType.GOAL_UPDATE,
            source="api.goals.adopt",
            handler=lambda: main_loop.adopt_goal(goal_id),
            payload={"goal_id": goal_id},
            correlation_id=goal_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(decision)


@router.post("/{goal_id}/transition")
def transition_goal(
    goal_id: str,
    body: GoalTransitionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    if body.status in {GoalStatus.CANDIDATE, GoalStatus.ACTIVE}:
        raise HTTPException(
            status_code=409,
            detail="Use goal adoption to activate or resume a goal",
        )
    main_loop = get_main_loop(request)
    try:
        goal = execute_agent_event(
            runtime,
            AgentEventType.GOAL_UPDATE,
            source="api.goals.transition",
            handler=lambda: main_loop.transition_goal(
                goal_id,
                body.status,
                reason=body.reason,
                outcome=body.outcome,
            ),
            payload={"goal_id": goal_id, "status": body.status.value},
            correlation_id=goal_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(goal)


@router.post("/reevaluate")
def reevaluate_goals(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    decisions = execute_agent_event(
        runtime,
        AgentEventType.GOAL_REEVALUATE,
        source="api.goals.reevaluate",
        handler=main_loop.reevaluate_goals,
    ).value
    return {"decisions": [asdict(decision) for decision in decisions]}


@commitment_router.get("")
def inspect_commitments(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = get_main_loop(request).commitment_store
    payload: dict[str, object] = {"commitments": store.to_json()}
    return execute_agent_event(
        runtime,
        AgentEventType.GOAL_READ,
        source="api.commitments.inspect",
        handler=lambda: payload,
    ).value


@commitment_router.post("")
def create_commitment(
    body: CommitmentRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        commitment = execute_agent_event(
            runtime,
            AgentEventType.GOAL_UPDATE,
            source="api.commitments.create",
            handler=lambda: main_loop.create_commitment(
                description=body.description,
                priority=body.priority,
                urgency=body.urgency,
                expected_utility=body.expected_utility,
                confidence=body.confidence,
                deadline=_deadline(body.deadline),
                value_effects=body.value_effects,
                conflict_ids=tuple(body.conflict_ids),
                commitment_id=body.commitment_id,
                origin_actor=OriginActor.OPERATOR,
                origin_source_ref="admin:commitment_request",
                interlocutor_key=body.interlocutor_key,
            ),
            payload={"commitment_id": body.commitment_id},
            correlation_id=body.commitment_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(commitment)


@commitment_router.post("/{commitment_id}/transition")
def transition_commitment(
    commitment_id: str,
    body: CommitmentTransitionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    if body.status == CommitmentStatus.ACTIVE:
        raise HTTPException(status_code=409, detail="Commitment is already active")
    main_loop = get_main_loop(request)
    try:
        commitment = execute_agent_event(
            runtime,
            AgentEventType.GOAL_UPDATE,
            source="api.commitments.transition",
            handler=lambda: main_loop.transition_commitment(
                commitment_id,
                body.status,
                reason=body.reason,
                outcome=body.outcome,
            ),
            payload={
                "commitment_id": commitment_id,
                "status": body.status.value,
            },
            correlation_id=commitment_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(commitment)


def _deadline(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()

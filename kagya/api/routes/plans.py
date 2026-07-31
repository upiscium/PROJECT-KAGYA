"""Strict Plan and Step lifecycle administration routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from collections.abc import Callable

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    get_private_operator,
    PrivateOperator,
)
from kagya.planning import EvidenceReference, Plan, PlanCandidate
from kagya.runtime import AgentEventType, AgentRuntime


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RevisionRequest(_RequestModel):
    expected_revision: int = Field(ge=1)
    reason_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    candidate: PlanCandidate


class EvidenceRequest(_RequestModel):
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)


class StepFailureRequest(_RequestModel):
    reason_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)


router = APIRouter(prefix="/api/plans", tags=["plans"])


@router.get("")
def inspect_plans(
    request: Request,
    goal_id: str | None = None,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = get_main_loop(request).plan_store
    plans = execute_agent_event(
        runtime,
        AgentEventType.PLAN_READ,
        source="api.plans.inspect",
        handler=lambda: store.list_plans(goal_id=goal_id),
    ).value
    return {"plans": [item.model_dump(mode="json") for item in plans]}


@router.get("/candidates")
def current_candidates(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    candidates = execute_agent_event(
        runtime,
        AgentEventType.PLAN_READ,
        source="api.plans.candidates",
        handler=main_loop.current_plan_candidates,
    ).value
    from dataclasses import asdict

    return {"candidates": [asdict(item) for item in candidates]}


@router.post("")
def create_plan(
    body: PlanCandidate,
    request: Request,
    operator: PrivateOperator = Depends(get_private_operator),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        plan = execute_agent_event(
            runtime,
            AgentEventType.PLAN_UPDATE,
            source="api.plans.create",
            handler=lambda: main_loop.create_plan(body, actor_id=operator.actor_id),
            payload={"plan_id": body.plan_id, "goal_id": body.goal_id},
            correlation_id=body.plan_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return plan.model_dump(mode="json")


@router.post("/{plan_id}/activate")
def activate_plan(
    plan_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute_plan_update(
        runtime,
        AgentEventType.PLAN_UPDATE,
        "api.plans.activate",
        plan_id,
        lambda: get_main_loop(request).activate_plan(plan_id),
    )


@router.post("/{plan_id}/revisions")
def revise_plan(
    plan_id: str,
    body: RevisionRequest,
    request: Request,
    operator: PrivateOperator = Depends(get_private_operator),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute_plan_update(
        runtime,
        AgentEventType.PLAN_REPLAN,
        "api.plans.revise",
        plan_id,
        lambda: get_main_loop(request).revise_plan(
            plan_id,
            body.candidate,
            expected_revision=body.expected_revision,
            reason_code=body.reason_code,
            actor_id=operator.actor_id,
        ),
    )


@router.post("/{plan_id}/steps/{step_id}/start")
def start_step(
    plan_id: str,
    step_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute_plan_update(
        runtime,
        AgentEventType.STEP_UPDATE,
        "api.plans.steps.start",
        plan_id,
        lambda: get_main_loop(request).start_plan_step(plan_id, step_id),
        step_id,
    )


@router.post("/{plan_id}/steps/{step_id}/complete")
def complete_step(
    plan_id: str,
    step_id: str,
    body: EvidenceRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute_plan_update(
        runtime,
        AgentEventType.STEP_UPDATE,
        "api.plans.steps.complete",
        plan_id,
        lambda: get_main_loop(request).complete_plan_step(
            plan_id, step_id, body.evidence
        ),
        step_id,
    )


@router.post("/{plan_id}/steps/{step_id}/fail")
def fail_step(
    plan_id: str,
    step_id: str,
    body: StepFailureRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute_plan_update(
        runtime,
        AgentEventType.STEP_UPDATE,
        "api.plans.steps.fail",
        plan_id,
        lambda: get_main_loop(request).fail_plan_step(
            plan_id,
            step_id,
            reason_code=body.reason_code,
            evidence=body.evidence,
        ),
        step_id,
    )


@router.post("/{plan_id}/abandon")
def abandon_plan(
    plan_id: str,
    body: EvidenceRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute_plan_update(
        runtime,
        AgentEventType.PLAN_UPDATE,
        "api.plans.abandon",
        plan_id,
        lambda: get_main_loop(request).abandon_plan(plan_id, body.evidence),
    )


def _execute_plan_update(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: str,
    plan_id: str,
    handler: Callable[[], Plan],
    step_id: str | None = None,
) -> dict[str, object]:
    try:
        plan = execute_agent_event(
            runtime,
            event_type,
            source=source,
            handler=handler,
            payload={"plan_id": plan_id, "step_id": step_id},
            correlation_id=plan_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return plan.model_dump(mode="json")

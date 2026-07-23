"""Operator API for governed action execution."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.actions import ActionBudget, ActionExecutionLayer, ActionPolicyError
from kagya.api.dependencies import (
    AdminActor,
    execute_agent_event,
    get_action_execution,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.runtime import AgentEventType, AgentRuntime


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntentRequest(_RequestModel):
    decision_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1, max_length=128)
    dry_run: bool = False
    budget: ActionBudget = Field(default_factory=ActionBudget)


class ApprovalRequest(_RequestModel):
    approved: bool
    reason: str | None = Field(default=None, max_length=500)


router = APIRouter(prefix="/api/actions", tags=["actions"])


@router.get("/intents", dependencies=[Depends(require_admin)])
def list_intents(
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    values = execute_agent_event(
        runtime,
        AgentEventType.ACTION_READ,
        source="api.actions.intents",
        handler=execution.list_intents,
    ).value
    return {"intents": [item.model_dump(mode="json") for item in values]}


@router.get("/approvals", dependencies=[Depends(require_admin)])
def approval_inbox(
    pending_only: bool = True,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    values = execute_agent_event(
        runtime,
        AgentEventType.ACTION_READ,
        source="api.actions.approvals",
        handler=lambda: execution.list_approvals(pending_only=pending_only),
    ).value
    return {"approvals": [item.model_dump(mode="json") for item in values]}


@router.get("/receipts", dependencies=[Depends(require_admin)])
def list_receipts(
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    receipts, observations = execute_agent_event(
        runtime,
        AgentEventType.ACTION_READ,
        source="api.actions.receipts",
        handler=lambda: (execution.list_receipts(), execution.list_observations()),
    ).value
    return {
        "receipts": [item.model_dump(mode="json") for item in receipts],
        "observations": [item.model_dump(mode="json") for item in observations],
    }


@router.post("/intents", dependencies=[Depends(require_admin)])
def create_intent(
    body: IntentRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _transition(
        runtime,
        AgentEventType.ACTION_INTENT,
        "api.actions.create",
        lambda: get_main_loop(request).action_coordinator.create_intent(
            body.decision_id,
            idempotency_key=body.idempotency_key,
            dry_run=body.dry_run,
            budget=body.budget,
        ),
        body.decision_id,
        {"decision_id": body.decision_id, "idempotency_key": body.idempotency_key},
    )


@router.post("/intents/{intent_id}/approval")
def resolve_approval(
    intent_id: str,
    body: ApprovalRequest,
    actor: AdminActor = Depends(require_admin),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _transition(
        runtime,
        AgentEventType.ACTION_APPROVAL,
        "api.actions.approval",
        lambda: execution.resolve_approval(
            intent_id,
            approved=body.approved,
            actor_id=actor.actor_id,
            reason=body.reason,
        ),
        intent_id,
        {"intent_id": intent_id, "approved": body.approved},
    )


@router.post("/intents/{intent_id}/execute", dependencies=[Depends(require_admin)])
def execute_intent(
    intent_id: str,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _transition(
        runtime,
        AgentEventType.ACTION_EXECUTE,
        "api.actions.execute",
        lambda: execution.execute(intent_id),
        intent_id,
        {"intent_id": intent_id},
    )


@router.post("/intents/{intent_id}/cancel", dependencies=[Depends(require_admin)])
def cancel_intent(
    intent_id: str,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _transition(
        runtime,
        AgentEventType.ACTION_CANCEL,
        "api.actions.cancel",
        lambda: execution.cancel(intent_id),
        intent_id,
        {"intent_id": intent_id},
    )


@router.post("/intents/{intent_id}/compensate", dependencies=[Depends(require_admin)])
def compensate_intent(
    intent_id: str,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _transition(
        runtime,
        AgentEventType.ACTION_COMPENSATE,
        "api.actions.compensate",
        lambda: execution.compensate(intent_id),
        intent_id,
        {"intent_id": intent_id},
    )


def _transition(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: str,
    handler: Callable[[], Any],
    correlation_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    try:
        value = execute_agent_event(
            runtime,
            event_type,
            source=source,
            handler=handler,
            payload=payload,
            correlation_id=correlation_id,
        ).value
    except (ValueError, ActionPolicyError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return value.model_dump(mode="json")

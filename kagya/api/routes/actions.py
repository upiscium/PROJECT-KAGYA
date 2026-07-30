"""Operator API for governed action execution."""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.actions import (
    ActionBudget,
    ActionExecutionLayer,
    ActionIntent,
    ActionPolicyError,
    ActionPolicyRejectionRecord,
    ActionValidationRecord,
    ApprovalRecord,
    ExecutionReceipt,
    IntentStatus,
    Observation,
    OutcomeVerification,
    ReceiptStatus,
    RiskClass,
    public_tool_name,
)
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


BoundedCode = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
]
PublicToolName = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
]
RecordT = TypeVar("RecordT")


class CockpitActionProvenanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision_id: str
    candidate_id: str
    triggering_event_id: str | None
    plan_id: str | None
    plan_revision: int | None = Field(default=None, ge=1)
    step_id: str | None


class CockpitActionApprovalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approval_id: str | None
    status: Literal["pending", "approved", "rejected"] | None
    requested_at: datetime | None
    resolved_at: datetime | None
    resolved_by_operator: bool


class CockpitActionReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    receipt_id: str
    status: ReceiptStatus
    attempt: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    event_id: str | None
    event_sequence: int | None = Field(default=None, ge=1)
    error_code: BoundedCode | None
    compensation_of: str | None


class CockpitActionRelatedReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    receipt_id: str
    status: ReceiptStatus


class CockpitActionObservationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observation_id: str
    valid: bool
    validation_errors: list[BoundedCode]
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class CockpitActionVerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verification_id: str
    success: bool
    reason: BoundedCode


class CockpitActionTraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    intent_id: str
    revision: int = Field(ge=1)
    tool_name: str
    risk_class: RiskClass
    status: IntentStatus
    dry_run: bool
    created_at: datetime
    updated_at: datetime
    failure_code: BoundedCode | None
    provenance: CockpitActionProvenanceResponse
    approval: CockpitActionApprovalResponse
    receipt: CockpitActionReceiptResponse | None
    related_receipts: list[CockpitActionRelatedReceiptResponse]
    observation: CockpitActionObservationResponse | None
    verification: CockpitActionVerificationResponse | None


class CockpitPreIntentFailureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    failure_id: str
    failure_type: Literal["validation", "policy_rejection"]
    decision_id: str | None
    candidate_id: str | None
    tool_name: PublicToolName | None
    risk_class: RiskClass | None
    error_codes: list[BoundedCode]
    event_id: str
    event_sequence: int = Field(ge=1)
    occurred_at: datetime


class CockpitActionTraceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pending_approval_count: int = Field(ge=0)
    retry_pending_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    traces: list[CockpitActionTraceResponse]
    pre_intent_failures: list[CockpitPreIntentFailureResponse]


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


@router.get(
    "/trace",
    response_model=CockpitActionTraceListResponse,
    dependencies=[Depends(require_admin)],
)
def cockpit_action_trace(
    limit: int = Query(default=50, ge=1, le=200),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> CockpitActionTraceListResponse:
    def build_trace() -> CockpitActionTraceListResponse:
        intents = execution.list_intents()
        validations = execution.list_validation_records()
        policy_rejections = execution.list_policy_rejections()
        approvals = _unique_by_id(
            execution.list_approvals(), lambda item: item.approval_id
        )
        receipts = _unique_by_id(
            execution.list_receipts(), lambda item: item.receipt_id
        )
        observations = _unique_by_id(
            execution.list_observations(), lambda item: item.observation_id
        )
        verifications = _unique_by_id(
            execution.list_verifications(), lambda item: item.verification_id
        )
        ordered = sorted(
            intents,
            key=lambda item: (item.updated_at, item.created_at, item.intent_id),
            reverse=True,
        )
        validation_by_id = _unique_by_id(
            validations, lambda item: item.validation_id
        )
        failed_validations = tuple(
            item for item in validation_by_id.values() if not item.arguments_valid
        )
        pre_intent_failures = sorted(
            [
                _validation_failure(item) for item in failed_validations
            ]
            + [
                _policy_rejection_failure(item, validation_by_id.get(item.validation_id))
                for item in policy_rejections
            ],
            key=lambda item: (item.occurred_at, item.failure_id),
            reverse=True,
        )
        return CockpitActionTraceListResponse(
            pending_approval_count=sum(
                item.status == IntentStatus.AWAITING_APPROVAL for item in intents
            ),
            retry_pending_count=sum(
                item.status == IntentStatus.RETRY_PENDING for item in intents
            ),
            failed_count=sum(
                item.status
                in {IntentStatus.FAILED, IntentStatus.REJECTED, IntentStatus.CANCELLED}
                for item in intents
            ) + len(failed_validations) + len(policy_rejections),
            traces=[
                _action_trace(
                    intent,
                    None
                    if intent.approval_id is None
                    else approvals.get(intent.approval_id),
                    None
                    if intent.receipt_id is None
                    else receipts.get(intent.receipt_id),
                    observations,
                    verifications,
                    receipts,
                )
                for intent in ordered[:limit]
            ],
            pre_intent_failures=pre_intent_failures[:limit],
        )

    return execute_agent_event(
        runtime,
        AgentEventType.ACTION_READ,
        source="api.actions.trace",
        handler=build_trace,
        payload={"limit": limit},
    ).value


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
    if isinstance(value, ActionValidationRecord):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "action_arguments_invalid",
                "validation_id": value.validation_id,
                "validation_error_codes": [
                    code.value for code in value.validation_error_codes
                ],
                "validation_schema_revision": value.validation_schema_revision,
                "canonical_arguments_digest": value.canonical_arguments_digest,
            },
        )
    return value.model_dump(mode="json")


def _action_trace(
    intent: ActionIntent,
    approval: ApprovalRecord | None,
    receipt: ExecutionReceipt | None,
    observations: dict[str, Observation],
    verifications: dict[str, OutcomeVerification],
    receipts: dict[str, ExecutionReceipt],
) -> CockpitActionTraceResponse:
    if approval is not None and approval.intent_id != intent.intent_id:
        approval = None
    if receipt is not None and not _receipt_matches_intent(receipt, intent):
        receipt = None
    observation = None
    verification = None
    if receipt is not None:
        candidate = (
            None
            if receipt.observation_id is None
            else observations.get(receipt.observation_id)
        )
        if (
            candidate is not None
            and candidate.intent_id == intent.intent_id
            and candidate.receipt_id == receipt.receipt_id
        ):
            observation = candidate
        candidate_verification = (
            None
            if receipt.verification_id is None
            else verifications.get(receipt.verification_id)
        )
        if (
            candidate_verification is not None
            and candidate_verification.intent_id == intent.intent_id
            and (
                candidate_verification.observation_id is None
                or (
                    observation is not None
                    and candidate_verification.observation_id
                    == observation.observation_id
                )
            )
        ):
            verification = candidate_verification
    return CockpitActionTraceResponse(
        intent_id=intent.intent_id,
        revision=intent.revision,
        tool_name=intent.tool_name,
        risk_class=intent.risk_class,
        status=intent.status,
        dry_run=intent.dry_run,
        created_at=intent.created_at,
        updated_at=intent.updated_at,
        failure_code=intent.failure_code,
        provenance=CockpitActionProvenanceResponse(
            decision_id=intent.provenance.decision_id,
            candidate_id=intent.provenance.candidate_id,
            triggering_event_id=intent.provenance.triggering_event_id,
            plan_id=intent.provenance.plan_id,
            plan_revision=intent.provenance.plan_revision,
            step_id=intent.provenance.step_id,
        ),
        approval=CockpitActionApprovalResponse(
            approval_id=None if approval is None else approval.approval_id,
            status=None if approval is None else approval.status,
            requested_at=None if approval is None else approval.requested_at,
            resolved_at=None if approval is None else approval.resolved_at,
            resolved_by_operator=(
                approval is not None
                and approval.resolved_at is not None
                and approval.actor_id is not None
            ),
        ),
        receipt=None if receipt is None else CockpitActionReceiptResponse(
            receipt_id=receipt.receipt_id,
            status=receipt.status,
            attempt=receipt.attempt,
            duration_ms=receipt.duration_ms,
            event_id=receipt.event_id,
            event_sequence=receipt.event_sequence,
            error_code=receipt.error_code,
            compensation_of=_safe_compensation_of(receipt, intent, receipts),
        ),
        related_receipts=[
            CockpitActionRelatedReceiptResponse(
                receipt_id=item.receipt_id,
                status=item.status,
            )
            for item in sorted(
                (
                    item
                    for item in receipts.values()
                    if _receipt_matches_intent(item, intent)
                    and (receipt is None or item.receipt_id != receipt.receipt_id)
                ),
                key=lambda item: (item.finished_at, item.receipt_id),
                reverse=True,
            )
        ],
        observation=None if observation is None else CockpitActionObservationResponse(
            observation_id=observation.observation_id,
            valid=observation.valid,
            validation_errors=list(observation.validation_errors),
            result_digest=observation.result_digest,
        ),
        verification=None if verification is None else CockpitActionVerificationResponse(
            verification_id=verification.verification_id,
            success=verification.success,
            reason=verification.reason,
        ),
    )


def _validation_failure(
    validation: ActionValidationRecord,
) -> CockpitPreIntentFailureResponse:
    return CockpitPreIntentFailureResponse(
        failure_id=validation.validation_id,
        failure_type="validation",
        decision_id=validation.decision_id,
        candidate_id=None,
        tool_name=public_tool_name(validation.tool_name),
        risk_class=validation.risk_class,
        error_codes=[code.value for code in validation.validation_error_codes],
        event_id=validation.validated_event_id,
        event_sequence=validation.validated_event_sequence,
        occurred_at=validation.validated_at,
    )


def _policy_rejection_failure(
    rejection: ActionPolicyRejectionRecord,
    validation: ActionValidationRecord | None,
) -> CockpitPreIntentFailureResponse:
    return CockpitPreIntentFailureResponse(
        failure_id=rejection.rejection_id,
        failure_type="policy_rejection",
        decision_id=rejection.decision_id,
        candidate_id=rejection.candidate_id,
        tool_name=(
            public_tool_name(validation.tool_name)
            if validation is not None
            and _validation_matches_rejection(validation, rejection)
            else None
        ),
        risk_class=rejection.risk_class,
        error_codes=[rejection.reason_code],
        event_id=rejection.event_id,
        event_sequence=rejection.event_sequence,
        occurred_at=rejection.rejected_at,
    )


def _receipt_matches_intent(
    receipt: ExecutionReceipt,
    intent: ActionIntent,
) -> bool:
    return (
        receipt.intent_id == intent.intent_id
        and receipt.idempotency_key == intent.idempotency_key
        and receipt.decision_id == intent.provenance.decision_id
        and receipt.plan_id == intent.provenance.plan_id
        and receipt.plan_revision == intent.provenance.plan_revision
        and receipt.step_id == intent.provenance.step_id
    )


def _safe_compensation_of(
    receipt: ExecutionReceipt,
    intent: ActionIntent,
    receipts: dict[str, ExecutionReceipt],
) -> str | None:
    if receipt.compensation_of is None:
        return None
    target = receipts.get(receipt.compensation_of)
    if (
        target is None
        or target.receipt_id == receipt.receipt_id
        or not _receipt_matches_intent(target, intent)
    ):
        return None
    return target.receipt_id


def _validation_matches_rejection(
    validation: ActionValidationRecord,
    rejection: ActionPolicyRejectionRecord,
) -> bool:
    return (
        validation.validation_id == rejection.validation_id
        and validation.arguments_valid is True
        and validation.intent_id is not None
        and validation.decision_id == rejection.decision_id
        and validation.risk_class == rejection.risk_class
    )


def _unique_by_id(
    values: tuple[RecordT, ...],
    identifier: Callable[[RecordT], str],
) -> dict[str, RecordT]:
    grouped: dict[str, list[RecordT]] = {}
    for item in values:
        grouped.setdefault(identifier(item), []).append(item)
    return {
        record_id: items[0]
        for record_id, items in grouped.items()
        if len(items) == 1
    }

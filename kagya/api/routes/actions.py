"""Operator API for governed action execution."""

from collections.abc import Callable
from datetime import datetime
from itertools import islice
import re
from typing import Annotated, Any, Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.actions import (
    ActionBudget,
    ActionExecutionLayer,
    ActionIntent,
    ActionOperatorSnapshot,
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
    OperatorCommand,
    OperatorCommandRequest,
)
from kagya.api.dependencies import (
    execute_agent_event,
    get_action_execution,
    get_agent_runtime,
    get_main_loop,
    get_private_operator,
    get_tool_registry,
    PrivateOperator,
)
from kagya.runtime import AgentEventType, AgentRuntime
from kagya.tools import ToolRegistry


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


class IntentCreatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["intent"]
    intent_id: str
    revision: int = Field(ge=1)
    status: IntentStatus


class IntentPolicyRejectedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["policy_rejection"]
    rejection_id: str
    policy_code: Literal["risk_budget_denied"]
    reason_code: Literal["risk_class_exceeds_budget"]


class OperatorMutationRequest(_RequestModel):
    expected_intent_revision: int = Field(ge=1)
    expected_preview_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_phrase: str | None = Field(default=None, max_length=200)


class OperatorApprovalRequest(OperatorMutationRequest):
    expected_approval_id: str
    approved: bool
    reason: str | None = Field(default=None, max_length=500)


BoundedCode = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
]


class ActionToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    risk_class: RiskClass
    approval_required: bool
    reversible: bool
    effect_code: str
    validation_schema_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    enabled: bool
    executable: bool
    execution_authority: Literal["action_execution"]


class RegistryToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    description: str | None
    tool_type: str
    status: str
    generated: bool
    human_approved: bool
    execution_authority: Literal["registry_only"]


class OperatorApprovalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    approval_id: str
    status: Literal["pending", "approved", "rejected"]
    requested_at: datetime


class MetadataArgumentSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["metadata_read"]
    namespace: str
    key: str


class DocumentArgumentSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["document_search"]
    scope_kind: Literal["all", "path"]
    max_results: int = Field(ge=1)
    query_length: int = Field(ge=0)


class CalendarArgumentSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["calendar_read"]
    starts_at: datetime
    ends_at: datetime
    max_results: int = Field(ge=1)


class NotificationArgumentSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["notification"]
    channel: Literal["local"]
    title: str = Field(min_length=1, max_length=120)
    body_preview: str = Field(min_length=1, max_length=1000)


OperatorArgumentSummaryResponse = (
    MetadataArgumentSummaryResponse
    | DocumentArgumentSummaryResponse
    | CalendarArgumentSummaryResponse
    | NotificationArgumentSummaryResponse
)


class OperatorPolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    allowed: bool
    approval_required: bool
    reason_codes: list[BoundedCode]


class OperatorPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    effect_code: BoundedCode
    effect: str = Field(min_length=1, max_length=240)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    compensation_available: bool


class OperatorBudgetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_attempts: int = Field(ge=1)
    max_cost_units: int = Field(ge=1)
    max_monetary_cost: int = Field(ge=0)
    deadline_at: datetime
    attempts: int = Field(ge=0)
    cost_units_used: int = Field(ge=0)
    retry_at: datetime | None


class OperatorProvenanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision_id: str
    plan_id: str | None
    plan_revision: int | None = Field(default=None, ge=1)
    step_id: str | None
    triggering_event_id: str | None


class OperatorReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    receipt_id: str
    status: ReceiptStatus


class OperatorVerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    verification_id: str
    success: bool
    reason: BoundedCode


class OperatorConfirmationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    required: Literal[True]
    phrase: str = Field(min_length=1, max_length=200)


class OperatorActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    intent_id: str
    revision: int
    status: IntentStatus
    approval: OperatorApprovalResponse | None
    tool: ActionToolResponse
    argument_summary: OperatorArgumentSummaryResponse
    policy: OperatorPolicyResponse
    preview: OperatorPreviewResponse
    budget: OperatorBudgetResponse
    provenance: OperatorProvenanceResponse
    receipt: OperatorReceiptResponse | None
    verification: OperatorVerificationResponse | None
    idempotency_state: Literal["reserved", "released", "completed", "unknown"]
    available_commands: list[OperatorCommand]
    confirmation: OperatorConfirmationResponse | None


class OperatorSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    pending_approval_count: int
    operator_action_count: int
    risk_ceiling: RiskClass
    actions: list[OperatorActionResponse]
    action_tools: list[ActionToolResponse]
    registry_tools: list[RegistryToolResponse]


class OperatorMutationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    command: OperatorCommand
    event_id: str
    processing_sequence: int = Field(ge=1)
    action: OperatorActionResponse
    disposition: Literal[
        "awaiting_scheduler", "rejected", "cancelled", "executed", "compensated"
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
REGISTRY_TOOLS_CAP = 100


@router.get("/intents")
def list_intents(
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    raise HTTPException(
        status_code=409, detail={"code": "operator_projection_required"}
    )


@router.get("/approvals")
def approval_inbox(
    pending_only: bool = True,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    raise HTTPException(
        status_code=409, detail={"code": "operator_projection_required"}
    )


@router.get("/receipts")
def list_receipts(
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    raise HTTPException(
        status_code=409, detail={"code": "operator_projection_required"}
    )


@router.get(
    "/trace",
    response_model=CockpitActionTraceListResponse,
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
        validation_by_id = _unique_by_id(validations, lambda item: item.validation_id)
        failed_validations = tuple(
            item for item in validation_by_id.values() if not item.arguments_valid
        )
        pre_intent_failures = sorted(
            [_validation_failure(item) for item in failed_validations]
            + [
                _policy_rejection_failure(
                    item, validation_by_id.get(item.validation_id)
                )
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
            )
            + len(failed_validations)
            + len(policy_rejections),
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


@router.get("/operator-summary", response_model=OperatorSummaryResponse)
def operator_summary(
    limit: int = Query(default=50, ge=1, le=200),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    registry: ToolRegistry = Depends(get_tool_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> OperatorSummaryResponse:
    def build() -> OperatorSummaryResponse:
        snapshot = execution.operator_snapshot()
        actions: list[OperatorActionResponse] = []
        for item in sorted(
            snapshot.intents.values(),
            key=lambda item: (item.updated_at, item.created_at, item.intent_id),
            reverse=True,
        ):
            try:
                projected = _operator_action(execution, item, snapshot)
            except (ValueError, ActionPolicyError):
                continue
            if projected.available_commands:
                actions.append(projected)
        descriptors = [_action_tool(item) for item in execution.list_tool_descriptors()]
        registry_tools = [
            RegistryToolResponse(
                name=item.name,
                description=None,
                tool_type=item.tool_type.value,
                status=item.status.value,
                generated=item.generated,
                human_approved=item.human_approved,
                execution_authority="registry_only",
            )
            for item in islice(registry.list(), REGISTRY_TOOLS_CAP)
            if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", item.name)
        ]
        enabled_risks = [item.risk_class for item in descriptors if item.enabled]
        ceiling = max(
            enabled_risks,
            default=RiskClass.READ_ONLY,
            key=lambda value: list(RiskClass).index(value),
        )
        return OperatorSummaryResponse(
            pending_approval_count=sum(
                item.status == IntentStatus.AWAITING_APPROVAL for item in actions
            ),
            operator_action_count=len(actions),
            risk_ceiling=ceiling,
            actions=actions[:limit],
            action_tools=descriptors,
            registry_tools=registry_tools,
        )

    return execute_agent_event(
        runtime,
        AgentEventType.ACTION_READ,
        source="api.actions.operator_summary",
        handler=build,
        payload={"limit": limit},
    ).value


@router.post(
    "/operator/intents/{intent_id}/approval", response_model=OperatorMutationResponse
)
def operator_approval(
    intent_id: str,
    body: OperatorApprovalRequest,
    operator: PrivateOperator = Depends(get_private_operator),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> OperatorMutationResponse:
    command = OperatorCommand.APPROVE if body.approved else OperatorCommand.REJECT
    request = OperatorCommandRequest(
        intent_id=intent_id,
        command=command,
        expected_revision=body.expected_intent_revision,
        expected_preview_digest=body.expected_preview_digest,
        expected_approval_id=body.expected_approval_id,
        confirmation=body.confirmation_phrase,
    )
    return _operator_command(execution, runtime, operator, request, body.reason)


@router.post(
    "/operator/intents/{intent_id}/cancel", response_model=OperatorMutationResponse
)
def operator_cancel(
    intent_id: str,
    body: OperatorMutationRequest,
    operator: PrivateOperator = Depends(get_private_operator),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> OperatorMutationResponse:
    return _operator_command(
        execution,
        runtime,
        operator,
        OperatorCommandRequest(
            intent_id=intent_id,
            command=OperatorCommand.CANCEL,
            expected_revision=body.expected_intent_revision,
            expected_preview_digest=body.expected_preview_digest,
            confirmation=body.confirmation_phrase,
        ),
    )


@router.post(
    "/operator/intents/{intent_id}/retry", response_model=OperatorMutationResponse
)
def operator_retry(
    intent_id: str,
    body: OperatorMutationRequest,
    operator: PrivateOperator = Depends(get_private_operator),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> OperatorMutationResponse:
    return _operator_command(
        execution,
        runtime,
        operator,
        OperatorCommandRequest(
            intent_id=intent_id,
            command=OperatorCommand.RETRY_NOW,
            expected_revision=body.expected_intent_revision,
            expected_preview_digest=body.expected_preview_digest,
            confirmation=body.confirmation_phrase,
        ),
    )


@router.post(
    "/operator/intents/{intent_id}/compensate", response_model=OperatorMutationResponse
)
def operator_compensate(
    intent_id: str,
    body: OperatorMutationRequest,
    operator: PrivateOperator = Depends(get_private_operator),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> OperatorMutationResponse:
    return _operator_command(
        execution,
        runtime,
        operator,
        OperatorCommandRequest(
            intent_id=intent_id,
            command=OperatorCommand.COMPENSATE,
            expected_revision=body.expected_intent_revision,
            expected_preview_digest=body.expected_preview_digest,
            confirmation=body.confirmation_phrase,
        ),
    )


@router.post(
    "/intents",
    response_model=IntentCreatedResponse | IntentPolicyRejectedResponse,
)
def create_intent(
    body: IntentRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> IntentCreatedResponse | IntentPolicyRejectedResponse:
    return _create_intent_transition(
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
    operator: PrivateOperator = Depends(get_private_operator),
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    raise HTTPException(status_code=409, detail={"code": "operator_preview_required"})


@router.post("/intents/{intent_id}/execute")
def execute_intent(
    intent_id: str,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    raise HTTPException(status_code=409, detail={"code": "operator_command_required"})


@router.post("/intents/{intent_id}/cancel")
def cancel_intent(
    intent_id: str,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    raise HTTPException(status_code=409, detail={"code": "operator_command_required"})


@router.post("/intents/{intent_id}/compensate")
def compensate_intent(
    intent_id: str,
    execution: ActionExecutionLayer = Depends(get_action_execution),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    raise HTTPException(status_code=409, detail={"code": "operator_command_required"})


def _create_intent_transition(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: str,
    handler: Callable[[], Any],
    correlation_id: str,
    payload: dict[str, object],
) -> IntentCreatedResponse | IntentPolicyRejectedResponse:
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
    if isinstance(value, ActionPolicyRejectionRecord):
        return IntentPolicyRejectedResponse(
            kind="policy_rejection",
            rejection_id=value.rejection_id,
            policy_code=value.policy_code,
            reason_code=value.reason_code,
        )
    if not isinstance(value, ActionIntent):
        raise HTTPException(status_code=409, detail={"code": "action_creation_failed"})
    return IntentCreatedResponse(
        kind="intent",
        intent_id=value.intent_id,
        revision=value.revision,
        status=value.status,
    )


def _action_tool(descriptor: Any) -> ActionToolResponse:
    return ActionToolResponse(
        name=descriptor.tool_name,
        risk_class=descriptor.risk_class,
        approval_required=descriptor.approval_required,
        reversible=descriptor.reversible,
        effect_code=descriptor.effect_code,
        validation_schema_revision=descriptor.validation_revision,
        enabled=descriptor.enabled,
        executable=descriptor.executable,
        execution_authority="action_execution",
    )


def _operator_action(
    execution: ActionExecutionLayer,
    intent: ActionIntent,
    snapshot: ActionOperatorSnapshot | None = None,
) -> OperatorActionResponse:
    preview = (
        execution.operator_preview_from_snapshot(snapshot, intent)
        if snapshot is not None
        else execution.operator_preview(intent.intent_id)
    )
    state_approvals = (
        snapshot.approvals
        if snapshot is not None
        else _unique_by_id(execution.list_approvals(), lambda item: item.approval_id)
    )
    approval = state_approvals.get(intent.approval_id) if intent.approval_id else None
    if approval is not None and approval.intent_id != intent.intent_id:
        approval = None
    receipts = (
        snapshot.receipts
        if snapshot is not None
        else _unique_by_id(execution.list_receipts(), lambda item: item.receipt_id)
    )
    receipt = receipts.get(intent.receipt_id) if intent.receipt_id else None
    if receipt is not None and not _receipt_matches_intent(receipt, intent):
        receipt = None
    verification = None
    if receipt is not None and receipt.verification_id:
        candidate = (
            snapshot.verifications
            if snapshot is not None
            else _unique_by_id(
                execution.list_verifications(), lambda item: item.verification_id
            )
        ).get(receipt.verification_id)
        observations = (
            snapshot.observations
            if snapshot is not None
            else _unique_by_id(
                execution.list_observations(), lambda item: item.observation_id
            )
        )
        observation = (
            None
            if receipt.observation_id is None
            else observations.get(receipt.observation_id)
        )
        if (
            candidate is not None
            and observation is not None
            and candidate.intent_id == intent.intent_id
            and candidate.observation_id == observation.observation_id
            and observation.intent_id == intent.intent_id
            and observation.receipt_id == receipt.receipt_id
        ):
            verification = candidate
    argument = _operator_argument_summary(preview.arguments)
    return OperatorActionResponse(
        intent_id=intent.intent_id,
        revision=intent.revision,
        status=intent.status,
        approval=None
        if approval is None
        else OperatorApprovalResponse(
            approval_id=approval.approval_id,
            status=approval.status,
            requested_at=approval.requested_at,
        ),
        tool=_action_tool(preview.tool),
        argument_summary=argument,
        policy=OperatorPolicyResponse(
            allowed=intent.policy.allowed,
            approval_required=intent.policy.approval_required,
            reason_codes=list(intent.policy.reasons),
        ),
        preview=OperatorPreviewResponse(
            effect_code=preview.effect_code,
            effect=preview.effect,
            digest=preview.preview_digest,
            compensation_available=intent.preview.compensation_available,
        ),
        budget=OperatorBudgetResponse(
            max_attempts=intent.budget.max_attempts,
            max_cost_units=intent.budget.max_cost_units,
            max_monetary_cost=intent.budget.max_monetary_cost,
            deadline_at=intent.deadline_at,
            attempts=intent.attempts,
            cost_units_used=intent.cost_units_used,
            retry_at=intent.retry_at,
        ),
        provenance=OperatorProvenanceResponse(
            decision_id=intent.provenance.decision_id,
            plan_id=intent.provenance.plan_id,
            plan_revision=intent.provenance.plan_revision,
            step_id=intent.provenance.step_id,
            triggering_event_id=intent.provenance.triggering_event_id,
        ),
        receipt=None
        if receipt is None
        else OperatorReceiptResponse(
            receipt_id=receipt.receipt_id, status=receipt.status
        ),
        verification=None
        if verification is None
        else OperatorVerificationResponse(
            verification_id=verification.verification_id,
            success=verification.success,
            reason=verification.reason,
        ),
        idempotency_state=(
            "completed"
            if intent.status
            in {
                IntentStatus.SUCCEEDED,
                IntentStatus.FAILED,
                IntentStatus.CANCELLED,
                IntentStatus.REJECTED,
                IntentStatus.COMPENSATED,
            }
            else "reserved"
        ),
        available_commands=list(
            execution.available_commands_from_snapshot(snapshot, intent)
            if snapshot is not None
            else execution.available_commands(intent.intent_id)
        ),
        confirmation=(
            None
            if intent.risk_class not in {RiskClass.DESTRUCTIVE, RiskClass.HIGH_IMPACT}
            else OperatorConfirmationResponse(
                required=True,
                phrase=f"CONFIRM {intent.intent_id} {intent.revision} {preview.preview_digest}",
            )
        ),
    )


def _operator_argument_summary(value: Any) -> OperatorArgumentSummaryResponse:
    data = value.model_dump(mode="json")
    kind = data.get("kind")
    if kind == "metadata":
        return MetadataArgumentSummaryResponse(
            kind="metadata_read",
            namespace=str(data["namespace"]),
            key=str(data["key"]),
        )
    if kind == "document_search":
        return DocumentArgumentSummaryResponse(
            kind="document_search",
            scope_kind="path" if data["path_scoped"] else "all",
            max_results=int(data["max_results"]),
            query_length=int(data["query_length"]),
        )
    if kind == "calendar_read":
        return CalendarArgumentSummaryResponse(
            kind="calendar_read",
            starts_at=datetime.fromisoformat(str(data["starts_at"])),
            ends_at=datetime.fromisoformat(str(data["ends_at"])),
            max_results=int(data["max_results"]),
        )
    if kind == "local_notification_enqueue":
        return NotificationArgumentSummaryResponse(
            kind="notification",
            channel="local",
            title=str(data["title"]),
            body_preview=str(data["body"]),
        )
    raise ActionPolicyError("Action public preview is unavailable")


def _operator_command(
    execution: ActionExecutionLayer,
    runtime: AgentRuntime,
    operator: PrivateOperator,
    request: OperatorCommandRequest,
    reason: str | None = None,
) -> OperatorMutationResponse:
    try:
        event_type = {
            OperatorCommand.APPROVE: AgentEventType.ACTION_APPROVAL,
            OperatorCommand.REJECT: AgentEventType.ACTION_APPROVAL,
            OperatorCommand.CANCEL: AgentEventType.ACTION_CANCEL,
            OperatorCommand.RETRY_NOW: AgentEventType.ACTION_EXECUTE,
            OperatorCommand.COMPENSATE: AgentEventType.ACTION_COMPENSATE,
        }[request.command]

        def apply() -> OperatorActionResponse:
            checked = execution.validate_command_request(request)
            if not checked.valid:
                raise _command_error(checked.reason)
            if request.command in {OperatorCommand.APPROVE, OperatorCommand.REJECT}:
                updated = execution.resolve_approval(
                    request.intent_id,
                    approved=request.command == OperatorCommand.APPROVE,
                    actor_id=operator.actor_id,
                    reason=reason,
                )
            else:
                updated = execution.execute_command(request, actor_id=operator.actor_id)
            return _operator_action(execution, updated)

        outcome = execute_agent_event(
            runtime,
            event_type,
            source=f"api.actions.operator.{request.command.value}",
            handler=apply,
            payload={"intent_id": request.intent_id, "command": request.command.value},
            correlation_id=request.intent_id,
        )
    except HTTPException:
        raise
    except _OperatorCommandFailure as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code}) from exc
    except (ValueError, ActionPolicyError) as exc:
        raise HTTPException(
            status_code=409, detail={"code": _command_code(str(exc))}
        ) from exc
    dispositions: dict[
        OperatorCommand,
        Literal[
            "awaiting_scheduler", "rejected", "cancelled", "executed", "compensated"
        ],
    ] = {
        OperatorCommand.APPROVE: "awaiting_scheduler",
        OperatorCommand.REJECT: "rejected",
        OperatorCommand.CANCEL: "cancelled",
        OperatorCommand.RETRY_NOW: "executed",
        OperatorCommand.COMPENSATE: "compensated",
    }
    return OperatorMutationResponse(
        command=request.command,
        event_id=outcome.event.event_id,
        processing_sequence=outcome.event.processing_sequence or 0,
        action=outcome.value,
        disposition=dispositions[request.command],
    )


class _OperatorCommandFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


def _command_error(reason: str | None) -> _OperatorCommandFailure:
    return _OperatorCommandFailure(_command_code(reason or ""))


def _command_code(reason: str) -> str:
    if "revision" in reason:
        return "stale_action_revision"
    if "digest" in reason:
        return "stale_preview_digest"
    if "approval" in reason:
        return "stale_approval_binding"
    if "confirmation" in reason:
        return "confirmation_required"
    if "available" in reason:
        return "command_unavailable"
    return "action_binding_invalid"


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
        receipt=None
        if receipt is None
        else CockpitActionReceiptResponse(
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
        observation=None
        if observation is None
        else CockpitActionObservationResponse(
            observation_id=observation.observation_id,
            valid=observation.valid,
            validation_errors=list(observation.validation_errors),
            result_digest=observation.result_digest,
        ),
        verification=None
        if verification is None
        else CockpitActionVerificationResponse(
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
        record_id: items[0] for record_id, items in grouped.items() if len(items) == 1
    }

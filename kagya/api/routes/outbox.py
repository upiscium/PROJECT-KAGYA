"""Admin API for proactive local delivery and correlated responses."""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_outbox,
    get_private_operator,
    PrivateOperator,
)
from kagya.outbox import (
    AcknowledgmentStatus,
    DeliveryStatus,
    Outbox,
    OutboxMessage,
    OutboxMessageKind,
    OutboxReferences,
    OutboxUrgency,
    PrivacyClass,
)
from kagya.runtime import AgentEventType, AgentRuntime, current_agent_event


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnqueueRequest(_Request):
    kind: OutboxMessageKind
    title: str = Field(min_length=1, max_length=160)
    body: str = Field(min_length=1, max_length=4000)
    public_preview: str | None = Field(default=None, min_length=1, max_length=160)
    deduplication_key: str = Field(min_length=1, max_length=256)
    context_id: str | None = None
    interlocutor_id: str | None = None
    references: OutboxReferences = Field(default_factory=OutboxReferences)
    urgency: OutboxUrgency = OutboxUrgency.NORMAL
    not_before: datetime | None = None
    expires_at: datetime | None = None
    privacy_class: PrivacyClass = PrivacyClass.OPERATOR

    @model_validator(mode="after")
    def validate_conversational_preview(self) -> "EnqueueRequest":
        if self.kind in {
            OutboxMessageKind.QUESTION,
            OutboxMessageKind.RENEGOTIATION,
        }:
            if self.public_preview is None:
                raise ValueError(
                    "Conversational outbox messages require a public preview"
                )
            if self.body != self.public_preview:
                raise ValueError("Conversational public preview must match body")
        return self


class ResponseRequest(_Request):
    kind: Literal["read", "reply", "approval", "reject"]
    text: str | None = Field(default=None, max_length=4000)


class FailureRequest(_Request):
    failure_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"
    )


class CockpitOutboxReferencesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: str | None
    goal_id: str | None
    plan_id: str | None
    decision_id: str | None
    action_id: str | None
    commitment_id: str | None


class CockpitOutboxMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    message_id: str
    title: str
    urgency: OutboxUrgency
    delivery_status: DeliveryStatus
    acknowledgment_status: AcknowledgmentStatus
    references: CockpitOutboxReferencesResponse


class SafeOutboxMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_id: str
    kind: Literal[
        "question",
        "approval_request",
        "commitment_deadline",
        "goal_state",
        "action_result",
        "anomaly",
        "renegotiation",
        "long_task_complete",
    ]
    title: str
    urgency: OutboxUrgency
    delivery_status: DeliveryStatus
    acknowledgment_status: AcknowledgmentStatus
    created_at: datetime
    channel: Literal["local"]
    privacy_class: PrivacyClass
    last_failure_code: str | None
    body_preview: str | None = Field(default=None, max_length=160)
    references: CockpitOutboxReferencesResponse


class SafeOutboxListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    messages: list[SafeOutboxMessageResponse]


class CockpitOutboxSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pending_count: int = Field(ge=0)
    critical_count: int = Field(ge=0)
    messages: list[CockpitOutboxMessageResponse]


router = APIRouter(prefix="/api/outbox", tags=["outbox"])


@router.get("/messages", response_model=SafeOutboxListResponse)
def list_messages(
    outbox: Outbox = Depends(get_outbox),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SafeOutboxListResponse:
    messages = execute_agent_event(
        runtime,
        AgentEventType.OUTBOX_READ,
        source="api.outbox.list",
        handler=outbox.list_messages,
    ).value
    return SafeOutboxListResponse(messages=[_safe_message(item) for item in messages])


@router.get(
    "/summary",
    response_model=CockpitOutboxSummaryResponse,
)
def cockpit_summary(
    limit: int = Query(default=50, ge=1, le=200),
    outbox: Outbox = Depends(get_outbox),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> CockpitOutboxSummaryResponse:
    def build_summary() -> CockpitOutboxSummaryResponse:
        messages = outbox.list_messages()
        return CockpitOutboxSummaryResponse(
            pending_count=sum(
                message.delivery_status == DeliveryStatus.PENDING
                for message in messages
            ),
            critical_count=sum(
                message.urgency == OutboxUrgency.CRITICAL for message in messages
            ),
            messages=[_cockpit_message(message) for message in messages[:limit]],
        )

    return execute_agent_event(
        runtime,
        AgentEventType.OUTBOX_READ,
        source="api.outbox.summary",
        handler=build_summary,
        payload={"limit": limit},
    ).value


@router.post("/messages", response_model=SafeOutboxMessageResponse)
def enqueue_message(
    body: EnqueueRequest,
    outbox: Outbox = Depends(get_outbox),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SafeOutboxMessageResponse:
    try:
        message = execute_agent_event(
            runtime,
            AgentEventType.OUTBOX_ENQUEUE,
            source="api.outbox.enqueue",
            handler=lambda: outbox.enqueue(
                body.kind,
                title=body.title,
                body=body.body,
                public_preview=body.public_preview,
                deduplication_key=body.deduplication_key,
                context_id=body.context_id,
                interlocutor_id=body.interlocutor_id,
                references=body.references,
                urgency=body.urgency,
                not_before=body.not_before,
                expires_at=body.expires_at,
                privacy_class=body.privacy_class,
            ),
            payload={
                "kind": body.kind.value,
                "deduplication_key": body.deduplication_key,
            },
            correlation_id=body.deduplication_key,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _safe_message(message)


@router.post("/deliveries", response_model=SafeOutboxListResponse)
def deliver_messages(
    limit: int = 20,
    outbox: Outbox = Depends(get_outbox),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SafeOutboxListResponse:
    try:
        messages = execute_agent_event(
            runtime,
            AgentEventType.OUTBOX_DELIVER,
            source="api.outbox.deliver",
            handler=lambda: outbox.deliver(limit=limit),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SafeOutboxListResponse(messages=[_safe_message(item) for item in messages])


@router.post(
    "/messages/{message_id}/responses", response_model=SafeOutboxMessageResponse
)
def respond_to_message(
    message_id: str,
    body: ResponseRequest,
    operator: PrivateOperator = Depends(get_private_operator),
    outbox: Outbox = Depends(get_outbox),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SafeOutboxMessageResponse:
    if body.kind in {"approval", "reject"}:
        raise HTTPException(
            status_code=409, detail={"code": "approval_requires_cockpit_preview"}
        )

    def correlate() -> OutboxMessage:
        event = current_agent_event()
        return outbox.respond(
            message_id,
            kind=body.kind,
            actor_id=operator.actor_id,
            text=body.text,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )

    try:
        message = execute_agent_event(
            runtime,
            AgentEventType.OUTBOX_RESPONSE,
            source="api.outbox.response",
            handler=correlate,
            payload={"message_id": message_id, "response_kind": body.kind},
            correlation_id=message_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _safe_message(message)


@router.post(
    "/messages/{message_id}/delivery-failures",
    response_model=SafeOutboxMessageResponse,
)
def record_delivery_failure(
    message_id: str,
    body: FailureRequest,
    outbox: Outbox = Depends(get_outbox),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SafeOutboxMessageResponse:
    try:
        message = execute_agent_event(
            runtime,
            AgentEventType.OUTBOX_FAILURE,
            source="api.outbox.failure",
            handler=lambda: outbox.fail_delivery(message_id, body.failure_code),
            payload={"message_id": message_id, "failure_code": body.failure_code},
            correlation_id=message_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _safe_message(message)


def _cockpit_message(message: OutboxMessage) -> CockpitOutboxMessageResponse:
    return CockpitOutboxMessageResponse(
        message_id=message.message_id,
        title=_operator_title(message.kind),
        urgency=message.urgency,
        delivery_status=message.delivery_status,
        acknowledgment_status=message.acknowledgment_status,
        references=CockpitOutboxReferencesResponse(
            event_id=message.references.event_id,
            goal_id=message.references.goal_id,
            plan_id=message.references.plan_id,
            decision_id=message.references.decision_id,
            action_id=message.references.action_id,
            commitment_id=message.references.commitment_id,
        ),
    )


def _safe_message(message: OutboxMessage) -> SafeOutboxMessageResponse:
    return SafeOutboxMessageResponse(
        message_id=message.message_id,
        kind=message.kind.value,
        title=_operator_title(message.kind),
        urgency=message.urgency,
        delivery_status=message.delivery_status,
        acknowledgment_status=message.acknowledgment_status,
        created_at=message.created_at,
        channel="local",
        privacy_class=message.privacy_class,
        last_failure_code=message.last_failure_code,
        body_preview=_public_body_preview(message),
        references=CockpitOutboxReferencesResponse(
            event_id=message.references.event_id,
            goal_id=message.references.goal_id,
            plan_id=message.references.plan_id,
            decision_id=message.references.decision_id,
            action_id=message.references.action_id,
            commitment_id=message.references.commitment_id,
        ),
    )


def _public_body_preview(message: OutboxMessage) -> str | None:
    """Return only the separately validated public conversational preview."""
    if message.kind not in {
        OutboxMessageKind.QUESTION,
        OutboxMessageKind.RENEGOTIATION,
    }:
        return None
    return message.public_preview


def _operator_title(kind: OutboxMessageKind) -> str:
    return {
        OutboxMessageKind.QUESTION: "Question",
        OutboxMessageKind.APPROVAL_REQUEST: "Action approval required",
        OutboxMessageKind.COMMITMENT_DEADLINE: "Commitment deadline",
        OutboxMessageKind.GOAL_STATE: "Goal state update",
        OutboxMessageKind.ACTION_RESULT: "Action result",
        OutboxMessageKind.ANOMALY: "Runtime anomaly",
        OutboxMessageKind.RENEGOTIATION: "Renegotiation requested",
        OutboxMessageKind.LONG_TASK_COMPLETE: "Long task completed",
    }[kind]

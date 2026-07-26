"""Structured feedback submission, revision, withdrawal, and audit routes."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.api.dependencies import (
    AdminActor,
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.feedback import FeedbackSignal, FeedbackTarget, FeedbackTargetType
from kagya.runtime import AgentEventType, AgentRuntime
from kagya.identity import SocialPressureMetadata
from kagya.api.routes._identity_boundary import retain_pressure_observation


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackTargetRequest(_RequestModel):
    target_type: FeedbackTargetType
    target_id: str = Field(min_length=1)
    episode_id: str | None = Field(default=None, min_length=1)
    experience_id: str | None = Field(default=None, min_length=1)
    decision_id: str | None = Field(default=None, min_length=1)
    context_id: str | None = Field(default=None, min_length=1)

    def to_domain(self) -> FeedbackTarget:
        return FeedbackTarget(**self.model_dump())


class FeedbackCreateRequest(_RequestModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    feedback_id: str | None = Field(default=None, min_length=1, max_length=200)
    target: FeedbackTargetRequest
    signals: list[FeedbackSignal] = Field(min_length=1)
    correction: str | None = Field(default=None, min_length=1, max_length=20000)
    expected_answer: str | None = Field(default=None, min_length=1, max_length=20000)

    @model_validator(mode="after")
    def validate_public_target(self) -> "FeedbackCreateRequest":
        if self.target.target_type not in {
            FeedbackTargetType.RESPONSE,
            FeedbackTargetType.EPISODE,
        }:
            raise ValueError("Public feedback must target a response or episode")
        if self.target.episode_id not in {None, self.target.target_id}:
            raise ValueError("Public feedback target and episode IDs must match")
        if self.target.decision_id is not None:
            raise ValueError("Public feedback cannot attach an arbitrary decision")
        if self.target.context_id is None:
            raise ValueError("Public response feedback requires its context ID")
        return self


class AdminFeedbackCreateRequest(FeedbackCreateRequest):
    social_pressure: SocialPressureMetadata | None = None

    @model_validator(mode="after")
    def validate_public_target(self) -> "AdminFeedbackCreateRequest":
        return self


class FeedbackRevisionRequest(_RequestModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    expected_revision: int = Field(ge=1)
    signals: list[FeedbackSignal] = Field(min_length=1)
    correction: str | None = Field(default=None, min_length=1, max_length=20000)
    expected_answer: str | None = Field(default=None, min_length=1, max_length=20000)


class FeedbackWithdrawalRequest(_RequestModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    expected_revision: int = Field(ge=1)


router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("")
def submit_feedback(
    body: FeedbackCreateRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.FEEDBACK_UPDATE,
            source="api.feedback.user",
            handler=lambda: main_loop.submit_feedback(
                target=body.target.to_domain(),
                signals=tuple(body.signals),
                idempotency_key=body.idempotency_key,
                actor_type="user",
                actor_id=None,
                source="api.feedback.user",
                correction=body.correction,
                expected_answer=body.expected_answer,
                feedback_id=body.feedback_id,
            ),
            payload={
                "target_type": body.target.target_type.value,
                "target_id": body.target.target_id,
                "signals": [item.value for item in body.signals],
            },
            correlation_id=body.target.context_id or body.target.target_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(record)


@router.post("/admin")
def submit_admin_feedback(
    body: AdminFeedbackCreateRequest,
    request: Request,
    actor: AdminActor = Depends(require_admin),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.FEEDBACK_UPDATE,
            source="api.feedback.operator",
            handler=lambda: retain_pressure_observation(
                main_loop,
                main_loop.submit_feedback(
                    target=body.target.to_domain(),
                    signals=tuple(body.signals),
                    idempotency_key=body.idempotency_key,
                    actor_type="operator",
                    actor_id=actor.actor_id,
                    source="api.feedback.operator",
                    correction=body.correction,
                    expected_answer=body.expected_answer,
                    feedback_id=body.feedback_id,
                ),
                body.social_pressure,
                context_id=body.target.context_id,
            ),
            payload={
                "target_type": body.target.target_type.value,
                "target_id": body.target.target_id,
                "signals": [item.value for item in body.signals],
            },
            correlation_id=body.target.context_id or body.target.target_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(record)


@router.get("", dependencies=[Depends(require_admin)])
def inspect_feedback(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    records = execute_agent_event(
        runtime,
        AgentEventType.FEEDBACK_READ,
        source="api.feedback.inspect",
        handler=get_main_loop(request).feedback_store.list_records,
    ).value
    return {"feedback": [asdict(record) for record in records]}


@router.get("/{feedback_id}", dependencies=[Depends(require_admin)])
def inspect_feedback_record(
    feedback_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.FEEDBACK_READ,
            source="api.feedback.inspect_record",
            handler=lambda: get_main_loop(request).feedback_store.get(feedback_id),
            correlation_id=feedback_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(record)


@router.post("/{feedback_id}/revisions")
def revise_feedback(
    feedback_id: str,
    body: FeedbackRevisionRequest,
    request: Request,
    actor: AdminActor = Depends(require_admin),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.FEEDBACK_UPDATE,
            source="api.feedback.revise",
            handler=lambda: get_main_loop(request).revise_feedback(
                feedback_id,
                expected_revision=body.expected_revision,
                signals=tuple(body.signals),
                idempotency_key=body.idempotency_key,
                actor_type="operator",
                actor_id=actor.actor_id,
                source="api.feedback.revise",
                correction=body.correction,
                expected_answer=body.expected_answer,
            ),
            payload={
                "feedback_id": feedback_id,
                "expected_revision": body.expected_revision,
                "signals": [item.value for item in body.signals],
            },
            correlation_id=feedback_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(record)


@router.post("/{feedback_id}/withdraw")
def withdraw_feedback(
    feedback_id: str,
    body: FeedbackWithdrawalRequest,
    request: Request,
    actor: AdminActor = Depends(require_admin),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.FEEDBACK_UPDATE,
            source="api.feedback.withdraw",
            handler=lambda: get_main_loop(request).withdraw_feedback(
                feedback_id,
                expected_revision=body.expected_revision,
                idempotency_key=body.idempotency_key,
                actor_type="operator",
                actor_id=actor.actor_id,
                source="api.feedback.withdraw",
            ),
            payload={
                "feedback_id": feedback_id,
                "expected_revision": body.expected_revision,
            },
            correlation_id=feedback_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(record)

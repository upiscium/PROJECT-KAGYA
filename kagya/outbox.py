"""Persistent, privacy-bounded proactive message outbox."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


OUTBOX_STATE_KEY = "proactive_outbox"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutboxMessageKind(StrEnum):
    QUESTION = "question"
    APPROVAL_REQUEST = "approval_request"
    COMMITMENT_DEADLINE = "commitment_deadline"
    GOAL_STATE = "goal_state"
    ACTION_RESULT = "action_result"
    ANOMALY = "anomaly"
    RENEGOTIATION = "renegotiation"
    LONG_TASK_COMPLETE = "long_task_complete"


class OutboxUrgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class OutboxChannel(StrEnum):
    LOCAL = "local"


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    INTERLOCUTOR = "interlocutor"
    OPERATOR = "operator"
    PRIVATE = "private"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class AcknowledgmentStatus(StrEnum):
    UNACKNOWLEDGED = "unacknowledged"
    READ = "read"
    REPLIED = "replied"
    APPROVED = "approved"
    REJECTED = "rejected"


class OutboxReferences(_StrictModel):
    event_id: str | None = None
    goal_id: str | None = None
    plan_id: str | None = None
    decision_id: str | None = None
    action_id: str | None = None
    commitment_id: str | None = None


class DeliveryAttempt(_StrictModel):
    attempt: int = Field(ge=1)
    attempted_at: datetime
    status: Literal["delivered", "failed"]
    failure_code: str | None = Field(
        default=None, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"
    )


class OutboxResponse(_StrictModel):
    response_id: str
    kind: Literal["read", "reply", "approval", "reject"]
    actor_id: str
    received_at: datetime
    text: str | None = Field(default=None, max_length=4000)
    event_id: str | None = None
    event_sequence: int | None = Field(default=None, ge=1)


class OutboxMessage(_StrictModel):
    schema_version: Literal[1] = 1
    message_id: str
    revision: int = Field(default=1, ge=1)
    kind: OutboxMessageKind
    title: str = Field(min_length=1, max_length=160)
    body: str = Field(min_length=1, max_length=4000)
    context_id: str | None = None
    interlocutor_id: str | None = None
    references: OutboxReferences = Field(default_factory=OutboxReferences)
    urgency: OutboxUrgency = OutboxUrgency.NORMAL
    not_before: datetime
    expires_at: datetime | None = None
    channel: OutboxChannel = OutboxChannel.LOCAL
    privacy_class: PrivacyClass = PrivacyClass.OPERATOR
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    acknowledgment_status: AcknowledgmentStatus = AcknowledgmentStatus.UNACKNOWLEDGED
    deduplication_key: str = Field(min_length=1, max_length=256)
    created_at: datetime
    updated_at: datetime
    delivered_at: datetime | None = None
    acknowledged_at: datetime | None = None
    attempts: tuple[DeliveryAttempt, ...] = ()
    responses: tuple[OutboxResponse, ...] = ()
    last_failure_code: str | None = Field(
        default=None, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"
    )

    @model_validator(mode="after")
    def validate_public_envelope(self) -> "OutboxMessage":
        timestamps = (
            self.not_before,
            self.expires_at,
            self.created_at,
            self.updated_at,
            self.delivered_at,
            self.acknowledged_at,
        )
        if any(value is not None and value.tzinfo is None for value in timestamps):
            raise ValueError("Outbox timestamps must include a timezone")
        if self.expires_at is not None and self.expires_at <= self.not_before:
            raise ValueError("Outbox expiry must be after not_before")
        if self.privacy_class == PrivacyClass.PRIVATE:
            raise ValueError("Private subject state cannot enter the outbox")
        if _contains_private_data(self.model_dump(mode="json")):
            raise ValueError("Outbox message contains a forbidden private field")
        return self


class OutboxState(_StrictModel):
    schema_version: Literal[1] = 1
    messages: tuple[OutboxMessage, ...] = ()


class Outbox:
    """Own outbox state and enforce delivery policy inside AgentRuntime events."""

    def __init__(
        self,
        main_loop: Any,
        *,
        quiet_hours_start: int = 22,
        quiet_hours_end: int = 7,
        max_deliveries_per_hour: int = 12,
        clock: Callable[[], datetime] | None = None,
        event_recorder: Any | None = None,
    ) -> None:
        if not 0 <= quiet_hours_start <= 23 or not 0 <= quiet_hours_end <= 23:
            raise ValueError("Quiet-hour boundaries must be valid hours")
        if max_deliveries_per_hour <= 0:
            raise ValueError("Outbox rate limit must be positive")
        self.main_loop = main_loop
        self.quiet_hours_start = quiet_hours_start
        self.quiet_hours_end = quiet_hours_end
        self.max_deliveries_per_hour = max_deliveries_per_hour
        self.clock = clock or (lambda: datetime.now(UTC))
        self.event_recorder = event_recorder
        self._state()

    def list_messages(self) -> tuple[OutboxMessage, ...]:
        return tuple(sorted(self._state().messages, key=lambda item: item.created_at, reverse=True))

    def get(self, message_id: str) -> OutboxMessage:
        message = next((item for item in self._state().messages if item.message_id == message_id), None)
        if message is None:
            raise ValueError(f"Unknown outbox message: {message_id}")
        return message

    def enqueue(
        self,
        kind: OutboxMessageKind,
        *,
        title: str,
        body: str,
        deduplication_key: str,
        context_id: str | None = None,
        interlocutor_id: str | None = None,
        references: OutboxReferences | None = None,
        urgency: OutboxUrgency = OutboxUrgency.NORMAL,
        not_before: datetime | None = None,
        expires_at: datetime | None = None,
        channel: OutboxChannel = OutboxChannel.LOCAL,
        privacy_class: PrivacyClass = PrivacyClass.OPERATOR,
    ) -> OutboxMessage:
        state = self._state()
        duplicate = next(
            (item for item in state.messages if item.deduplication_key == deduplication_key),
            None,
        )
        if duplicate is not None:
            return duplicate
        now = self.clock()
        message = OutboxMessage(
            message_id=str(uuid4()),
            kind=kind,
            title=title,
            body=body,
            context_id=context_id,
            interlocutor_id=interlocutor_id,
            references=references or OutboxReferences(),
            urgency=urgency,
            not_before=not_before or now,
            expires_at=expires_at,
            channel=channel,
            privacy_class=privacy_class,
            deduplication_key=deduplication_key,
            created_at=now,
            updated_at=now,
        )
        self._save(state.model_copy(update={"messages": (*state.messages, message)}))
        return message

    def deliver(self, *, limit: int = 20, now: datetime | None = None) -> tuple[OutboxMessage, ...]:
        if limit <= 0 or limit > 100:
            raise ValueError("Delivery limit must be between 1 and 100")
        current = now or self.clock()
        self._expire(current)
        state = self._state()
        recent = sum(
            item.delivered_at is not None
            and item.delivered_at > current - timedelta(hours=1)
            for item in state.messages
        )
        available = max(0, self.max_deliveries_per_hour - recent)
        candidates = [
            item
            for item in sorted(state.messages, key=lambda value: (-_urgency(value.urgency), value.created_at))
            if item.delivery_status in {DeliveryStatus.PENDING, DeliveryStatus.FAILED}
            and item.acknowledgment_status == AcknowledgmentStatus.UNACKNOWLEDGED
            and item.not_before <= current
            and (item.expires_at is None or item.expires_at > current)
            and (item.urgency == OutboxUrgency.CRITICAL or not self._quiet(current))
        ][: min(limit, available)]
        if not candidates:
            return ()
        selected = {item.message_id for item in candidates}
        updated = tuple(
            item.model_copy(
                update={
                    "revision": item.revision + 1,
                    "delivery_status": DeliveryStatus.DELIVERED,
                    "delivered_at": current,
                    "updated_at": current,
                    "last_failure_code": None,
                    "attempts": (*item.attempts, DeliveryAttempt(
                        attempt=len(item.attempts) + 1,
                        attempted_at=current,
                        status="delivered",
                    )),
                }
            )
            if item.message_id in selected
            else item
            for item in state.messages
        )
        self._save(state.model_copy(update={"messages": updated}))
        return tuple(self.get(item.message_id) for item in candidates)

    def fail_delivery(self, message_id: str, failure_code: str) -> OutboxMessage:
        message = self.get(message_id)
        if message.delivery_status in {
            DeliveryStatus.EXPIRED,
            DeliveryStatus.CANCELLED,
        } or message.acknowledgment_status != AcknowledgmentStatus.UNACKNOWLEDGED:
            return message
        now = self.clock()
        updated = message.model_copy(update={
            "revision": message.revision + 1,
            "delivery_status": DeliveryStatus.FAILED,
            "delivered_at": None,
            "updated_at": now,
            "last_failure_code": failure_code,
            "attempts": (*message.attempts, DeliveryAttempt(
                attempt=len(message.attempts) + 1,
                attempted_at=now,
                status="failed",
                failure_code=failure_code,
            )),
        })
        self._replace(updated)
        self._audit("delivery_failed", updated, failure_code=failure_code)
        return updated

    def cancel(self, deduplication_key: str) -> OutboxMessage | None:
        message = next(
            (
                item
                for item in self._state().messages
                if item.deduplication_key == deduplication_key
            ),
            None,
        )
        if (
            message is None
            or message.acknowledgment_status
            != AcknowledgmentStatus.UNACKNOWLEDGED
            or message.delivery_status
            in {DeliveryStatus.EXPIRED, DeliveryStatus.CANCELLED}
        ):
            return message
        now = self.clock()
        updated = message.model_copy(
            update={
                "revision": message.revision + 1,
                "delivery_status": DeliveryStatus.CANCELLED,
                "updated_at": now,
            }
        )
        self._replace(updated)
        return updated

    def respond(
        self,
        message_id: str,
        *,
        kind: Literal["read", "reply", "approval", "reject"],
        actor_id: str,
        text: str | None = None,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> OutboxMessage:
        message = self.get(message_id)
        if message.delivery_status == DeliveryStatus.EXPIRED:
            raise ValueError("Expired outbox messages cannot be acknowledged")
        requested = {
            "read": AcknowledgmentStatus.READ,
            "reply": AcknowledgmentStatus.REPLIED,
            "approval": AcknowledgmentStatus.APPROVED,
            "reject": AcknowledgmentStatus.REJECTED,
        }[kind]
        if message.acknowledgment_status != AcknowledgmentStatus.UNACKNOWLEDGED:
            if message.acknowledgment_status == requested:
                return message
            raise ValueError("Outbox message is already acknowledged")
        if kind == "reply" and not text:
            raise ValueError("Reply acknowledgment requires text")
        if kind in {"approval", "reject"} and message.kind != OutboxMessageKind.APPROVAL_REQUEST:
            raise ValueError("Only approval requests accept approval decisions")
        now = self.clock()
        response = OutboxResponse(
            response_id=str(uuid4()),
            kind=kind,
            actor_id=actor_id,
            text=text,
            received_at=now,
            event_id=event_id,
            event_sequence=event_sequence,
        )
        updated = message.model_copy(update={
            "revision": message.revision + 1,
            "acknowledgment_status": requested,
            "acknowledged_at": now,
            "updated_at": now,
            "responses": (*message.responses, response),
        })
        self._replace(updated)
        return updated

    def respond_to_action_approval(
        self,
        action_id: str,
        *,
        approved: bool,
        actor_id: str,
        text: str | None = None,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> OutboxMessage | None:
        matches = [
            item
            for item in self._state().messages
            if item.kind == OutboxMessageKind.APPROVAL_REQUEST
            and item.references.action_id == action_id
        ]
        if len(matches) != 1:
            return None
        message = matches[0]
        kind: Literal["approval", "reject"] = "approval" if approved else "reject"
        requested = AcknowledgmentStatus.APPROVED if approved else AcknowledgmentStatus.REJECTED
        if message.acknowledgment_status == requested:
            return message
        if message.acknowledgment_status in {
            AcknowledgmentStatus.APPROVED,
            AcknowledgmentStatus.REJECTED,
        }:
            return message
        # Outbox is a correlated notification projection, not approval authority.
        # A prior read/reply or delivery expiry must not veto an action transition
        # that already passed the preview-bound ActionExecution contract.
        now = self.clock()
        response = OutboxResponse(
            response_id=str(uuid4()),
            kind=kind,
            actor_id=actor_id,
            text=text,
            received_at=now,
            event_id=event_id,
            event_sequence=event_sequence,
        )
        updated = message.model_copy(update={
            "revision": message.revision + 1,
            "acknowledgment_status": requested,
            "acknowledged_at": now,
            "updated_at": now,
            "responses": (*message.responses, response),
        })
        self._replace(updated)
        return updated

    def _state(self) -> OutboxState:
        raw = self.main_loop.persistent_state.extensions.get(OUTBOX_STATE_KEY)
        if raw is None:
            state = OutboxState()
            self._save(state)
            return state
        try:
            return OutboxState.model_validate(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid proactive outbox state") from exc

    def _save(self, state: OutboxState) -> None:
        self.main_loop.persistent_state.extensions[OUTBOX_STATE_KEY] = state.model_dump(mode="json")

    def _replace(self, message: OutboxMessage) -> None:
        state = self._state()
        self._save(state.model_copy(update={
            "messages": tuple(message if item.message_id == message.message_id else item for item in state.messages)
        }))

    def _expire(self, now: datetime) -> None:
        state = self._state()
        changed = False
        messages: list[OutboxMessage] = []
        for item in state.messages:
            if (
                item.expires_at is not None
                and item.expires_at <= now
                and item.delivery_status
                not in {DeliveryStatus.EXPIRED, DeliveryStatus.CANCELLED}
                and item.acknowledgment_status == AcknowledgmentStatus.UNACKNOWLEDGED
            ):
                item = item.model_copy(update={
                    "revision": item.revision + 1,
                    "delivery_status": DeliveryStatus.EXPIRED,
                    "updated_at": now,
                })
                changed = True
            messages.append(item)
        if changed:
            self._save(state.model_copy(update={"messages": tuple(messages)}))

    def _quiet(self, now: datetime) -> bool:
        local = now.timetz().replace(tzinfo=None)
        start = time(self.quiet_hours_start)
        end = time(self.quiet_hours_end)
        if start == end:
            return False
        return start <= local or local < end if start > end else start <= local < end

    def _audit(self, event_type: str, message: OutboxMessage, **metadata: str) -> None:
        if self.event_recorder is None:
            return
        self.event_recorder.record(
            category="outbox",
            event_type=event_type,
            message="Outbox delivery lifecycle event",
            metadata={
                "message_id": message.message_id,
                "kind": message.kind.value,
                "deduplication_key": message.deduplication_key,
                **metadata,
            },
        )


def _urgency(value: OutboxUrgency) -> int:
    return {
        OutboxUrgency.LOW: 0,
        OutboxUrgency.NORMAL: 1,
        OutboxUrgency.HIGH: 2,
        OutboxUrgency.CRITICAL: 3,
    }[value]


def _contains_private_data(value: Any) -> bool:
    forbidden = {
        "hiddenthought",
        "privatestate",
        "prompt",
        "turns",
        "attachments",
        "eventpayload",
    }
    if isinstance(value, dict):
        return any(
            "".join(character for character in str(key).lower() if character.isalnum())
            in forbidden
            or _contains_private_data(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_data(item) for item in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return "<think>" in lowered or "</think>" in lowered
    return False


__all__ = [
    "AcknowledgmentStatus",
    "DeliveryStatus",
    "OUTBOX_STATE_KEY",
    "Outbox",
    "OutboxChannel",
    "OutboxMessage",
    "OutboxMessageKind",
    "OutboxReferences",
    "OutboxState",
    "OutboxUrgency",
    "PrivacyClass",
]

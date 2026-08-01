from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

import pytest

from kagya.outbox import (
    AcknowledgmentStatus,
    DeliveryStatus,
    OUTBOX_STATE_KEY,
    Outbox,
    OutboxMessageKind,
    OutboxReferences,
    OutboxUrgency,
    PrivacyClass,
)
from kagya.runtime import PersistentAgentState


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record(self, **values: object) -> None:
        self.records.append(values)


def _outbox(
    clock: _Clock,
    *,
    state: PersistentAgentState | None = None,
    rate: int = 12,
    recorder: _Recorder | None = None,
) -> Outbox:
    loop = SimpleNamespace(persistent_state=state or PersistentAgentState())
    return Outbox(
        loop,
        quiet_hours_start=22,
        quiet_hours_end=7,
        max_deliveries_per_hour=rate,
        clock=clock,
        event_recorder=recorder,
    )


def test_outbox_persists_references_and_deduplicates_across_restart() -> None:
    clock = _Clock(datetime(2026, 7, 23, 12, tzinfo=UTC))
    first = _outbox(clock)
    created = first.enqueue(
        OutboxMessageKind.QUESTION,
        title="Need input",
        body="Which bounded option should continue?",
        public_preview="Which bounded option should continue?",
        deduplication_key="goal-question:one",
        context_id="context-1",
        interlocutor_id="operator-1",
        references=OutboxReferences(
            event_id="event-1",
            goal_id="goal-1",
            plan_id="plan-1",
            decision_id="decision-1",
            action_id="action-1",
        ),
    )
    persisted = json.loads(json.dumps(first.main_loop.persistent_state.extensions))
    restarted_state = PersistentAgentState(extensions=persisted)
    restarted = _outbox(clock, state=restarted_state)

    duplicate = restarted.enqueue(
        OutboxMessageKind.QUESTION,
        title="Duplicate",
        body="This body must not replace the first message.",
        public_preview="This body must not replace the first message.",
        deduplication_key="goal-question:one",
    )

    assert duplicate.message_id == created.message_id
    assert duplicate.references.goal_id == "goal-1"
    assert len(restarted.list_messages()) == 1
    assert restarted_state.extensions[OUTBOX_STATE_KEY]["schema_version"] == 1


def test_delivery_enforces_quiet_hours_rate_expiry_and_ack_terminal_state() -> None:
    clock = _Clock(datetime(2026, 7, 23, 23, tzinfo=UTC))
    outbox = _outbox(clock, rate=1)
    normal = outbox.enqueue(
        OutboxMessageKind.GOAL_STATE,
        title="Goal updated",
        body="The goal changed state.",
        deduplication_key="goal:one",
    )
    critical = outbox.enqueue(
        OutboxMessageKind.ANOMALY,
        title="Critical anomaly",
        body="A bounded action requires immediate attention.",
        deduplication_key="anomaly:one",
        urgency=OutboxUrgency.CRITICAL,
    )
    expired = outbox.enqueue(
        OutboxMessageKind.ACTION_RESULT,
        title="Old result",
        body="This result is no longer actionable.",
        deduplication_key="expired:one",
        not_before=clock.value - timedelta(hours=2),
        expires_at=clock.value - timedelta(hours=1),
    )

    assert [item.message_id for item in outbox.deliver()] == [critical.message_id]
    assert outbox.get(expired.message_id).delivery_status == DeliveryStatus.EXPIRED
    assert outbox.get(normal.message_id).delivery_status == DeliveryStatus.PENDING
    acknowledged = outbox.respond(
        critical.message_id, kind="read", actor_id="operator-1"
    )
    assert acknowledged.acknowledgment_status == AcknowledgmentStatus.READ

    clock.value = datetime(2026, 7, 24, 0, tzinfo=UTC)
    assert outbox.deliver() == ()
    clock.value = datetime(2026, 7, 24, 12, tzinfo=UTC)
    assert [item.message_id for item in outbox.deliver()] == [normal.message_id]
    assert outbox.deliver() == ()


def test_private_content_is_rejected_and_failures_are_safely_audited() -> None:
    clock = _Clock(datetime(2026, 7, 23, 12, tzinfo=UTC))
    recorder = _Recorder()
    outbox = _outbox(clock, recorder=recorder)

    with pytest.raises(ValueError, match="Private subject state"):
        outbox.enqueue(
            OutboxMessageKind.ANOMALY,
            title="Private",
            body="Not deliverable",
            deduplication_key="private:one",
            privacy_class=PrivacyClass.PRIVATE,
        )
    with pytest.raises(ValueError, match="forbidden private"):
        outbox.enqueue(
            OutboxMessageKind.ANOMALY,
            title="Leaked thought",
            body="<think>private chain</think>",
            deduplication_key="private:two",
        )

    message = outbox.enqueue(
        OutboxMessageKind.ACTION_RESULT,
        title="Result",
        body="A public result is ready.",
        deduplication_key="result:one",
    )
    failed = outbox.fail_delivery(message.message_id, "local_transport_unavailable")

    assert failed.delivery_status == DeliveryStatus.FAILED
    assert failed.attempts[-1].failure_code == "local_transport_unavailable"
    serialized_audit = json.dumps(recorder.records)
    assert "local_transport_unavailable" in serialized_audit
    assert "A public result is ready" not in serialized_audit


@pytest.mark.parametrize(
    "kind", [OutboxMessageKind.QUESTION, OutboxMessageKind.RENEGOTIATION]
)
def test_conversational_enqueue_requires_body_bound_public_preview(
    kind: OutboxMessageKind,
) -> None:
    outbox = _outbox(_Clock(datetime(2026, 7, 23, 12, tzinfo=UTC)))

    with pytest.raises(ValueError, match="require a public preview"):
        outbox.enqueue(
            kind,
            title="Need input",
            body="A conversational body.",
            deduplication_key=f"missing-preview:{kind}",
        )
    with pytest.raises(ValueError, match="must match body"):
        outbox.enqueue(
            kind,
            title="Need input",
            body="A conversational body.",
            public_preview="An unrelated preview.",
            deduplication_key=f"mismatched-preview:{kind}",
        )


def test_legacy_conversational_record_with_null_preview_remains_readable() -> None:
    timestamp = datetime(2026, 7, 23, 12, tzinfo=UTC)
    state = PersistentAgentState(
        extensions={
            OUTBOX_STATE_KEY: {
                "schema_version": 1,
                "messages": [
                    {
                        "schema_version": 1,
                        "message_id": "legacy-question",
                        "kind": "question",
                        "title": "Legacy question",
                        "body": "Legacy body remains operator-only.",
                        "public_preview": None,
                        "not_before": timestamp.isoformat(),
                        "deduplication_key": "legacy-question",
                        "created_at": timestamp.isoformat(),
                        "updated_at": timestamp.isoformat(),
                    }
                ],
            }
        }
    )

    message = _outbox(_Clock(timestamp), state=state).get("legacy-question")

    assert message.public_preview is None


def test_reply_preserves_authoritative_origin_and_is_idempotent() -> None:
    clock = _Clock(datetime(2026, 7, 23, 12, tzinfo=UTC))
    outbox = _outbox(clock)
    message = outbox.enqueue(
        OutboxMessageKind.QUESTION,
        title="Goal question",
        body="Provide the missing constraint.",
        public_preview="Provide the missing constraint.",
        deduplication_key="question:goal-2",
        references=OutboxReferences(goal_id="goal-2", decision_id="decision-2"),
    )
    outbox.deliver()
    replied = outbox.respond(
        message.message_id,
        kind="reply",
        actor_id="operator-1",
        text="Use the local-only option.",
        event_id="response-event",
        event_sequence=9,
    )
    duplicate = outbox.respond(
        message.message_id,
        kind="reply",
        actor_id="operator-1",
        text="Use the local-only option.",
    )

    assert duplicate.revision == replied.revision
    assert len(replied.responses) == 1
    assert replied.references.goal_id == "goal-2"
    assert replied.responses[0].event_sequence == 9
    assert outbox.deliver() == ()

"""Contract checks for the browser-safe operator projections."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kagya.api.routes.actions import OperatorMutationRequest, OperatorSummaryResponse
from kagya.api.routes.outbox import SafeOutboxMessageResponse
from kagya.outbox import (
    AcknowledgmentStatus,
    DeliveryStatus,
    OutboxUrgency,
    PrivacyClass,
)


def test_operator_request_and_summary_are_strict() -> None:
    with pytest.raises(ValidationError):
        OperatorMutationRequest(
            expected_intent_revision=1,
            expected_preview_digest="0" * 64,
            unexpected="private arguments",
        )
    with pytest.raises(ValidationError):
        OperatorSummaryResponse(
            pending_approval_count=0,
            operator_action_count=0,
            risk_ceiling="read_only",
            actions=[],
            action_tools=[],
            registry_tools=[],
            raw_arguments="must not cross boundary",
        )


def test_safe_outbox_projection_has_no_body_or_delivery_history() -> None:
    value = SafeOutboxMessageResponse(
        message_id="message-1",
        kind="action_result",
        title="Action result",
        urgency=OutboxUrgency.NORMAL,
        delivery_status=DeliveryStatus.DELIVERED,
        acknowledgment_status=AcknowledgmentStatus.UNACKNOWLEDGED,
        created_at=datetime.now(UTC),
        channel="local",
        privacy_class=PrivacyClass.OPERATOR,
        last_failure_code=None,
        references={
            "event_id": None,
            "goal_id": None,
            "plan_id": None,
            "decision_id": None,
            "action_id": "action-1",
            "commitment_id": None,
        },
    )
    assert "body" not in value.model_dump()
    assert "deduplication_key" not in value.model_dump()
    assert "attempts" not in value.model_dump()

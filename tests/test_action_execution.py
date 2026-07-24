from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kagya.actions.execution as action_execution
from kagya.actions import (
    ACTION_STATE_KEY,
    ActionBudget,
    ActionExecutionLayer,
    ActionPolicyError,
    ActionState,
    ActionValidationRecord,
    IntentStatus,
    ReceiptStatus,
)
from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionStatus,
    DecisionStore,
    PredictedOutcome,
)
from kagya.runtime import AgentEventType, AgentRuntime, PersistentAgentState
from kagya.outbox import DeliveryStatus, Outbox


class _Loop:
    def __init__(self) -> None:
        self.persistent_state = PersistentAgentState()
        self.decision_store = DecisionStore()
        self.settings = SimpleNamespace(
            project=SimpleNamespace(name="PROJECT-KAGYA", environment="test"),
            deployment=SimpleNamespace(node=SimpleNamespace(id="test-node")),
            model=SimpleNamespace(provider="dummy"),
        )

    def record_decision_outcome(
        self,
        decision_id: str,
        *,
        description: str,
        utility: float,
        success: bool,
    ) -> object:
        from kagya.runtime import current_agent_event

        event = current_agent_event()
        return self.decision_store.record_outcome(
            decision_id,
            description=description,
            utility=utility,
            success=success,
            observed_event_id=None if event is None else event.event_id,
            observed_event_sequence=None if event is None else event.processing_sequence,
        )

    def record_decision_compensation(
        self, decision_id: str, *, receipt_id: str
    ) -> object:
        return self.decision_store.record_compensation(
            decision_id,
            receipt_id=receipt_id,
            observed_event_id=None,
            observed_event_sequence=None,
        )


def test_read_action_flows_through_runtime_to_observation_and_decision(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    _decision(
        loop,
        "decision-read",
        "restricted_metadata_read",
        {"namespace": "project", "key": "name"},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    runtime = AgentRuntime(queue_capacity=8)
    runtime.start()
    try:
        intent = runtime.execute(
            AgentEventType.ACTION_INTENT,
            source="test.intent",
            handler=lambda: layer.create_from_decision(
                "decision-read", idempotency_key="read-1"
            ),
        ).value
        assert intent.status == IntentStatus.APPROVED
        assert intent.preview.risk_class.value == "read_only"

        completed = runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.execute",
            handler=lambda: layer.execute(intent.intent_id),
        ).value
        duplicate = runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.duplicate",
            handler=lambda: layer.execute(intent.intent_id),
        ).value
    finally:
        runtime.shutdown()

    assert completed.status == IntentStatus.SUCCEEDED
    assert duplicate.receipt_id == completed.receipt_id
    assert len(layer.list_receipts()) == 1
    receipt = layer.list_receipts()[0]
    observation = layer.list_observations()[0]
    assert receipt.event_id is not None
    assert receipt.event_sequence == 2
    assert receipt.observation_id == observation.observation_id
    assert observation.data == {
        "namespace": "project",
        "key": "name",
        "value": "PROJECT-KAGYA",
    }
    decision = loop.decision_store.get("decision-read")
    assert decision.status == DecisionStatus.RESOLVED
    assert decision.actual_outcome is not None and decision.actual_outcome.success
    assert decision.actual_outcome.observed_event_id == receipt.event_id


def test_notification_requires_approval_is_idempotent_and_compensates(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    loop.outbox = Outbox(loop, quiet_hours_start=0, quiet_hours_end=0)
    _decision(
        loop,
        "decision-notify",
        "local_notification_enqueue",
        {"channel": "local", "title": "Review", "body": "Action needed"},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    runtime = AgentRuntime(queue_capacity=8)
    runtime.start()
    intent = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.notification.intent",
        handler=lambda: layer.create_from_decision(
            "decision-notify", idempotency_key="notification-1"
        ),
    ).value
    assert not isinstance(intent, ActionValidationRecord)

    assert intent.status == IntentStatus.AWAITING_APPROVAL
    assert len(layer.list_approvals(pending_only=True)) == 1
    with pytest.raises(ActionPolicyError, match="not executable"):
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.notification.unapproved",
            handler=lambda: layer.execute(intent.intent_id),
        )

    approved = runtime.execute(
        AgentEventType.ACTION_APPROVAL,
        source="test.notification.approval",
        handler=lambda: layer.resolve_approval(
            intent.intent_id, approved=True, actor_id="operator-1"
        ),
    ).value
    completed = runtime.execute(
        AgentEventType.ACTION_EXECUTE,
        source="test.notification.execute",
        handler=lambda: layer.execute(approved.intent_id),
    ).value
    assert completed.status == IntentStatus.SUCCEEDED
    state = ActionState.model_validate(loop.persistent_state.extensions[ACTION_STATE_KEY])
    assert len(state.notifications) == 1
    assert state.notifications[0]["status"] == "queued"
    assert len(loop.outbox.list_messages()) == 2
    assert next(
        item for item in loop.outbox.list_messages() if item.kind.value == "approval_request"
    ).acknowledgment_status.value == "approved"

    restored = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    assert (
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.notification.duplicate",
            handler=lambda: restored.execute(completed.intent_id),
        ).value.receipt_id
        == completed.receipt_id
    )
    assert len(ActionState.model_validate(
        loop.persistent_state.extensions[ACTION_STATE_KEY]
    ).notifications) == 1

    compensated = restored.compensate(completed.intent_id)
    state = ActionState.model_validate(loop.persistent_state.extensions[ACTION_STATE_KEY])
    assert compensated.status == IntentStatus.COMPENSATED
    assert state.notifications[0]["status"] == "cancelled"
    assert next(
        item for item in loop.outbox.list_messages() if item.title == "Review"
    ).delivery_status == DeliveryStatus.CANCELLED
    assert state.receipts[-1].status == ReceiptStatus.COMPENSATED
    assert state.receipts[-1].compensation_of == completed.receipt_id
    outcome = loop.decision_store.get("decision-notify").actual_outcome
    assert outcome is not None and outcome.compensated
    assert outcome.compensation_receipt_id == compensated.receipt_id
    runtime.shutdown()


def test_invalid_tools_arguments_dry_run_and_corrupt_state_fail_closed(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    _decision(loop, "decision-shell", "shell", {"command": "id"})
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    runtime = AgentRuntime(queue_capacity=8)
    runtime.start()
    shell_rejection = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.invalid.shell",
        handler=lambda: layer.create_from_decision(
            "decision-shell", idempotency_key="shell-1"
        ),
    ).value
    assert isinstance(shell_rejection, ActionValidationRecord)
    assert shell_rejection.arguments_valid is False

    _decision(
        loop,
        "decision-invalid",
        "document_search",
        {"query": "x", "relative_path": "../secret"},
    )
    invalid = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.invalid.path",
        handler=lambda: layer.create_from_decision(
            "decision-invalid", idempotency_key="invalid-1"
        ),
    ).value
    assert isinstance(invalid, ActionValidationRecord)
    assert invalid.arguments_valid is False
    assert tuple(code.value for code in invalid.validation_error_codes) == (
        "argument_path_out_of_scope",
    )
    assert layer.list_intents() == ()
    assert layer.list_receipts() == ()

    _decision(
        loop,
        "decision-preview",
        "restricted_metadata_read",
        {"namespace": "runtime", "key": "node_id"},
    )
    preview = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.preview.intent",
        handler=lambda: layer.create_from_decision(
            "decision-preview", idempotency_key="preview-1", dry_run=True
        ),
    ).value
    assert not isinstance(preview, ActionValidationRecord)
    assert preview.status == IntentStatus.DRY_RUN
    assert not layer.list_receipts()
    with pytest.raises(ActionPolicyError, match="Dry-run"):
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.preview.execute",
            handler=lambda: layer.execute(preview.intent_id),
        )

    _decision(
        loop,
        "decision-collision",
        "restricted_metadata_read",
        {"namespace": "project", "key": "name"},
    )
    with pytest.raises(ActionPolicyError, match="another decision"):
        runtime.execute(
            AgentEventType.ACTION_INTENT,
            source="test.collision.intent",
            handler=lambda: layer.create_from_decision(
                "decision-collision", idempotency_key="preview-1"
            ),
        )

    loop.persistent_state.extensions[ACTION_STATE_KEY] = {"schema_version": 99}
    with pytest.raises(ValueError, match="Invalid action execution state"):
        ActionExecutionLayer(
            loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
        )
    runtime.shutdown()


def test_retry_is_bounded_and_cancellable(tmp_path: Path) -> None:
    loop = _Loop()
    _decision(
        loop,
        "decision-retry",
        "document_search",
        {"query": "anything", "max_results": 1},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    runtime = AgentRuntime(queue_capacity=8)
    runtime.start()
    intent = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.retry.intent",
        handler=lambda: layer.create_from_decision(
            "decision-retry",
            idempotency_key="retry-1",
            budget=ActionBudget(max_attempts=2),
        ),
    ).value
    assert not isinstance(intent, ActionValidationRecord)

    def unavailable(_: object, __: object) -> object:
        raise OSError("temporarily unavailable")

    layer._invoke = unavailable  # type: ignore[method-assign]
    retrying = runtime.execute(
        AgentEventType.ACTION_EXECUTE,
        source="test.retry.execute",
        handler=lambda: layer.execute(intent.intent_id),
    ).value
    assert retrying.status == IntentStatus.RETRY_PENDING
    assert retrying.attempts == 1
    cancelled = layer.cancel(intent.intent_id)
    assert cancelled.status == IntentStatus.CANCELLED
    assert loop.decision_store.get("decision-retry").actual_outcome is not None
    with pytest.raises(ActionPolicyError, match="not executable"):
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.retry.cancelled",
            handler=lambda: layer.execute(intent.intent_id),
        )
    runtime.shutdown()


def test_rejection_records_structured_verification_for_later_attribution(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    _decision(
        loop,
        "decision-rejected",
        "local_notification_enqueue",
        {"channel": "local", "title": "Review", "body": "Reject this"},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    runtime = AgentRuntime(queue_capacity=8)
    runtime.start()
    intent = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.rejected.intent",
        handler=lambda: layer.create_from_decision(
            "decision-rejected", idempotency_key="rejected-1"
        ),
    ).value
    assert not isinstance(intent, ActionValidationRecord)

    rejected = layer.resolve_approval(
        intent.intent_id,
        approved=False,
        actor_id="operator-1",
        reason="not_authorized",
    )

    receipt = layer.get_receipt(rejected.receipt_id or "")
    observation = layer.get_observation(receipt.observation_id or "")
    verification = layer.list_verifications()[0]
    assert receipt.status == ReceiptStatus.CANCELLED
    assert observation.data == {
        "status": "cancelled",
        "error_code": "operator_rejected",
    }
    assert verification.observation_id == observation.observation_id
    assert verification.success is False
    runtime.shutdown()


def test_action_validation_requires_event_and_missing_record_rejects_before_call(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    _decision(
        loop,
        "decision-missing-validation",
        "restricted_metadata_read",
        {"namespace": "project", "key": "name"},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    with pytest.raises(RuntimeError, match="authoritative AgentRuntime event"):
        layer.create_from_decision(
            "decision-missing-validation", idempotency_key="missing-validation"
        )

    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    intent = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.validation.missing.intent",
        handler=lambda: layer.create_from_decision(
            "decision-missing-validation", idempotency_key="missing-validation"
        ),
    ).value
    assert not isinstance(intent, ActionValidationRecord)
    with pytest.raises(ActionPolicyError, match="authoritative event"):
        layer.execute(intent.intent_id)
    state = ActionState.model_validate(loop.persistent_state.extensions[ACTION_STATE_KEY])
    missing = intent.model_copy(update={"validation_record_id": None})
    loop.persistent_state.extensions[ACTION_STATE_KEY] = state.model_copy(
        update={"intents": (missing,)}
    ).model_dump(mode="json")
    called = False

    def invoke(_: object, __: object) -> object:
        nonlocal called
        called = True
        return {}

    layer._invoke = invoke  # type: ignore[method-assign]
    with pytest.raises(ActionPolicyError, match="no validation record"):
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.validation.missing.execute",
            handler=lambda: layer.execute(intent.intent_id),
        )
    assert called is False
    assert layer.list_receipts() == ()
    runtime.shutdown()


def test_action_validation_rejects_argument_mutation_and_schema_change(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    _decision(
        loop,
        "decision-stale",
        "restricted_metadata_read",
        {"namespace": "project", "key": "name"},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    runtime = AgentRuntime(queue_capacity=8)
    runtime.start()
    intent = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.validation.stale.intent",
        handler=lambda: layer.create_from_decision(
            "decision-stale", idempotency_key="stale"
        ),
    ).value
    assert not isinstance(intent, ActionValidationRecord)
    state = ActionState.model_validate(loop.persistent_state.extensions[ACTION_STATE_KEY])
    tampered = intent.model_copy(
        update={"arguments": {"namespace": "project", "key": "environment"}}
    )
    loop.persistent_state.extensions[ACTION_STATE_KEY] = state.model_copy(
        update={"intents": (tampered,)}
    ).model_dump(mode="json")
    with pytest.raises(ActionPolicyError, match="changed after validation"):
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.validation.tampered.execute",
            handler=lambda: layer.execute(intent.intent_id),
        )
    assert layer.list_receipts() == ()

    loop.persistent_state.extensions[ACTION_STATE_KEY] = state.model_dump(mode="json")
    revision = action_execution._VALIDATION_SCHEMA_REVISIONS[
        "restricted_metadata_read"
    ]
    action_execution._VALIDATION_SCHEMA_REVISIONS["restricted_metadata_read"] += 1
    try:
        with pytest.raises(ActionPolicyError, match="schema revision is stale"):
            runtime.execute(
                AgentEventType.ACTION_EXECUTE,
                source="test.validation.schema-changed.execute",
                handler=lambda: layer.execute(intent.intent_id),
            )
    finally:
        action_execution._VALIDATION_SCHEMA_REVISIONS[
            "restricted_metadata_read"
        ] = revision
        runtime.shutdown()
    assert layer.list_receipts() == ()


def test_action_validation_rejects_record_binding_and_event_mismatch(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    _decision(
        loop,
        "decision-mismatch",
        "restricted_metadata_read",
        {"namespace": "project", "key": "name"},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    runtime = AgentRuntime(queue_capacity=8)
    runtime.start()
    intent = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.validation.mismatch.intent",
        handler=lambda: layer.create_from_decision(
            "decision-mismatch", idempotency_key="mismatch"
        ),
    ).value
    assert not isinstance(intent, ActionValidationRecord)
    state = ActionState.model_validate(loop.persistent_state.extensions[ACTION_STATE_KEY])
    record = state.validation_records[0]

    mismatched = record.model_copy(update={"intent_id": "another-intent"})
    loop.persistent_state.extensions[ACTION_STATE_KEY] = state.model_copy(
        update={"validation_records": (mismatched,)}
    ).model_dump(mode="json")
    with pytest.raises(ActionPolicyError, match="binding is inconsistent"):
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.validation.binding-mismatch.execute",
            handler=lambda: layer.execute(intent.intent_id),
        )
    assert layer.list_receipts() == ()

    future_event = record.model_copy(
        update={"validated_event_sequence": record.validated_event_sequence + 100}
    )
    loop.persistent_state.extensions[ACTION_STATE_KEY] = state.model_copy(
        update={"validation_records": (future_event,)}
    ).model_dump(mode="json")
    with pytest.raises(ActionPolicyError, match="event is inconsistent"):
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.validation.event-mismatch.execute",
            handler=lambda: layer.execute(intent.intent_id),
        )
    assert layer.list_receipts() == ()
    runtime.shutdown()


def _decision(
    loop: _Loop,
    decision_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    action = ActionCandidate(
        candidate_id=f"{decision_id}-action",
        candidate_type=ActionType.INTERNAL,
        proposed_action=f"Use {tool_name}",
        parameters={"action": {"tool_name": tool_name, "arguments": arguments}},
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome(
                outcome_id="success", description="Action succeeds", probability=1.0, utility=1.0
            ),
        ),
        uncertainty=0.0,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )
    fallback = ActionCandidate(
        candidate_id=f"{decision_id}-fallback",
        candidate_type=ActionType.NO_OP,
        proposed_action="Do nothing",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome(
                outcome_id="idle", description="No action", probability=1.0, utility=-1.0
            ),
        ),
        uncertainty=0.0,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )
    loop.decision_store.create(
        [action, fallback],
        triggering_event_id="event-source",
        triggering_event_sequence=7,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id=decision_id,
    )

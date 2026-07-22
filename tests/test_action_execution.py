from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kagya.actions import (
    ACTION_STATE_KEY,
    ActionBudget,
    ActionExecutionLayer,
    ActionPolicyError,
    ActionState,
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
    _decision(
        loop,
        "decision-notify",
        "local_notification_enqueue",
        {"channel": "local", "title": "Review", "body": "Action needed"},
    )
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    intent = layer.create_from_decision(
        "decision-notify", idempotency_key="notification-1"
    )

    assert intent.status == IntentStatus.AWAITING_APPROVAL
    assert len(layer.list_approvals(pending_only=True)) == 1
    with pytest.raises(ActionPolicyError, match="not executable"):
        layer.execute(intent.intent_id)

    approved = layer.resolve_approval(
        intent.intent_id, approved=True, actor_id="operator-1"
    )
    completed = layer.execute(approved.intent_id)
    assert completed.status == IntentStatus.SUCCEEDED
    state = ActionState.model_validate(loop.persistent_state.extensions[ACTION_STATE_KEY])
    assert len(state.notifications) == 1
    assert state.notifications[0]["status"] == "queued"

    restored = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    assert restored.execute(completed.intent_id).receipt_id == completed.receipt_id
    assert len(ActionState.model_validate(
        loop.persistent_state.extensions[ACTION_STATE_KEY]
    ).notifications) == 1

    compensated = restored.compensate(completed.intent_id)
    state = ActionState.model_validate(loop.persistent_state.extensions[ACTION_STATE_KEY])
    assert compensated.status == IntentStatus.COMPENSATED
    assert state.notifications[0]["status"] == "cancelled"
    assert state.receipts[-1].status == ReceiptStatus.COMPENSATED
    assert state.receipts[-1].compensation_of == completed.receipt_id
    outcome = loop.decision_store.get("decision-notify").actual_outcome
    assert outcome is not None and outcome.compensated
    assert outcome.compensation_receipt_id == compensated.receipt_id


def test_invalid_tools_arguments_dry_run_and_corrupt_state_fail_closed(
    tmp_path: Path,
) -> None:
    loop = _Loop()
    _decision(loop, "decision-shell", "shell", {"command": "id"})
    layer = ActionExecutionLayer(
        loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
    )
    with pytest.raises(ActionPolicyError, match="not allowlisted"):
        layer.create_from_decision("decision-shell", idempotency_key="shell-1")

    _decision(
        loop,
        "decision-invalid",
        "document_search",
        {"query": "x", "relative_path": "../secret"},
    )
    with pytest.raises(ActionPolicyError, match="strict schema"):
        layer.create_from_decision("decision-invalid", idempotency_key="invalid-1")

    _decision(
        loop,
        "decision-preview",
        "restricted_metadata_read",
        {"namespace": "runtime", "key": "node_id"},
    )
    preview = layer.create_from_decision(
        "decision-preview", idempotency_key="preview-1", dry_run=True
    )
    assert preview.status == IntentStatus.DRY_RUN
    assert not layer.list_receipts()
    with pytest.raises(ActionPolicyError, match="Dry-run"):
        layer.execute(preview.intent_id)

    _decision(
        loop,
        "decision-collision",
        "restricted_metadata_read",
        {"namespace": "project", "key": "name"},
    )
    with pytest.raises(ActionPolicyError, match="another decision"):
        layer.create_from_decision(
            "decision-collision", idempotency_key="preview-1"
        )

    loop.persistent_state.extensions[ACTION_STATE_KEY] = {"schema_version": 99}
    with pytest.raises(ValueError, match="Invalid action execution state"):
        ActionExecutionLayer(
            loop, document_root=tmp_path, calendar_path=tmp_path / "calendar.json"
        )


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
    intent = layer.create_from_decision(
        "decision-retry",
        idempotency_key="retry-1",
        budget=ActionBudget(max_attempts=2),
    )

    def unavailable(_: object, __: object) -> object:
        raise OSError("temporarily unavailable")

    layer._invoke = unavailable  # type: ignore[method-assign]
    retrying = layer.execute(intent.intent_id)
    assert retrying.status == IntentStatus.RETRY_PENDING
    assert retrying.attempts == 1
    cancelled = layer.cancel(intent.intent_id)
    assert cancelled.status == IntentStatus.CANCELLED
    assert loop.decision_store.get("decision-retry").actual_outcome is not None
    with pytest.raises(ActionPolicyError, match="not executable"):
        layer.execute(intent.intent_id)


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

import json

import pytest

from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionDatasetGenerator,
    DecisionStore,
)
from kagya.feedback import (
    FeedbackPropagation,
    FeedbackProvenance,
    FeedbackSignal,
    FeedbackStore,
    FeedbackTarget,
    FeedbackTargetType,
    TrainingDisposition,
    feedback_fingerprint,
    normalize_signals,
)


def test_feedback_store_round_trip_and_idempotency_conflict() -> None:
    store = FeedbackStore()
    target = FeedbackTarget(FeedbackTargetType.EPISODE, "episode-1")
    provenance = FeedbackProvenance(
        actor_type="user",
        actor_id=None,
        source="test",
        event_id="event-1",
        event_sequence=1,
        submitted_at="2026-01-01T00:00:00+00:00",
    )
    propagation = FeedbackPropagation(
        memory_id="episode-1",
        correction_memory_id=None,
        memory_before={"lifecycle_status": "active"},
        memory_after={"lifecycle_status": "active"},
        decision_id=None,
        decision_outcome_applied=False,
        prediction_error=None,
        value_evidence=None,
        training_disposition=TrainingDisposition.INCLUDE,
        exclusion_refs=(),
        reason_codes=("good",),
    )
    fingerprint = feedback_fingerprint("create", {"signals": ["good"]})
    created = store.create(
        signals=(FeedbackSignal.GOOD,),
        target=target,
        provenance=provenance,
        correction_memory_id=None,
        expected_answer_memory_id=None,
        propagation=propagation,
        idempotency_key="operation-1",
        fingerprint=fingerprint,
        feedback_id="feedback-1",
    )

    assert store.idempotent_result("operation-1", fingerprint) == created
    with pytest.raises(ValueError, match="another operation"):
        store.idempotent_result("operation-1", "different")
    restored = FeedbackStore()
    restored.restore(json.loads(json.dumps(store.to_json())))
    assert restored.get("feedback-1") == created


def test_feedback_signal_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        normalize_signals((FeedbackSignal.GOOD, FeedbackSignal.BAD))
    with pytest.raises(ValueError, match="mutually exclusive"):
        normalize_signals((FeedbackSignal.REMEMBER, FeedbackSignal.DO_NOT_REMEMBER))


def test_decision_feedback_outcome_and_exclusion_are_reversible() -> None:
    store = DecisionStore()
    candidate = ActionCandidate(
        candidate_id="defer",
        candidate_type=ActionType.DEFER,
        proposed_action="Defer",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(),
        uncertainty=0.0,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )
    store.create(
        [candidate],
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id="decision-1",
    )

    resolved = store.record_feedback_outcome(
        "decision-1",
        utility=-0.75,
        success=False,
        feedback_id="feedback-1",
        feedback_revision=1,
        observed_event_id="event-1",
        observed_event_sequence=1,
    )
    excluded = store.set_training_policy(
        "decision-1", included=False, feedback_id="feedback-1"
    )

    assert resolved.prediction_error == pytest.approx(-0.75)
    assert resolved.actual_outcome is not None
    assert resolved.actual_outcome.description == "explicit_structured_feedback"
    assert excluded.training_included is False
    assert DecisionDatasetGenerator().generate([excluded]) == []
    awaiting = store.withdraw_feedback_outcome("decision-1", feedback_id="feedback-1")
    included = store.set_training_policy(
        "decision-1", included=True, feedback_id="feedback-1"
    )
    assert awaiting.actual_outcome is None
    assert included.training_included is True

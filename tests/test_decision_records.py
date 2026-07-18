import json

import pytest

from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionDatasetGenerator,
    DecisionStatus,
    DecisionStore,
    PredictedOutcome,
    parse_candidate_output,
    schema_candidate_prompt,
)


def test_candidate_comparison_tracks_contributions_and_selects_best() -> None:
    store = DecisionStore()

    record = store.create(
        [_respond_candidate(), _fallback_candidate()],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id="context-1",
        active_goal_ids=("goal-1",),
        value_revision_refs={"honesty": 2},
        emotion_snapshot={"valence": 0.1, "arousal": 0.2, "optimal_loss": 1.0},
        value_evaluator=lambda options: {
            "respond": {"honesty": 0.4},
            "no-op": {},
        },
        decision_id="decision-1",
        adapter_id="adapter-1",
        adapter_hash="hash-1",
        activation_sequence=7,
    )

    assert record.selected_candidate_id == "respond"
    assert record.status == DecisionStatus.AWAITING_OUTCOME
    assert record.actual_outcome is None
    assert record.prediction_error is None
    selected = record.considered_candidates[0]
    assert selected.value_contributions == {"honesty": 0.4}
    assert selected.appraisal_contributions == {"goal_progress": 0.3}
    assert selected.total_score is not None
    assert record.active_goal_ids == ("goal-1",)
    assert record.adapter_id == "adapter-1"
    assert record.adapter_hash == "hash-1"
    assert record.activation_sequence == 7
    assert record.schema_version == 7


def test_no_op_defer_and_observation_are_regular_candidates() -> None:
    for action_type in (
        ActionType.NO_OP,
        ActionType.DEFER,
        ActionType.OBSERVE,
        ActionType.REQUEST_INFORMATION,
    ):
        store = DecisionStore()
        fallback = _fallback_candidate(action_type=action_type)

        record = store.create(
            [fallback],
            triggering_event_id=None,
            triggering_event_sequence=None,
            context_id=None,
            active_goal_ids=(),
            value_revision_refs={},
            emotion_snapshot={},
        )

        assert record.selected_candidate_id == fallback.candidate_id
        assert record.considered_candidates[0].eligible is True


def test_missing_prerequisite_excludes_candidate_from_selection() -> None:
    store = DecisionStore()
    blocked = ActionCandidate(
        **{
            **_respond_candidate().__dict__,
            "prerequisites": ("goal:completed",),
        }
    )

    record = store.create(
        [blocked, _fallback_candidate()],
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
    )

    assert record.selected_candidate_id == "no-op"
    assert record.considered_candidates[0].eligible is False
    assert "goal:completed" in record.considered_candidates[0].reasons


def test_outcome_updates_same_record_and_computes_prediction_error() -> None:
    store = DecisionStore()
    record = store.create(
        [_respond_candidate(), _fallback_candidate()],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id="context-1",
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id="decision-1",
    )

    resolved = store.record_outcome(
        record.decision_id,
        description="Partial success",
        utility=0.2,
        success=True,
        observed_event_id="event-2",
        observed_event_sequence=2,
    )

    assert resolved.decision_id == record.decision_id
    assert resolved.status == DecisionStatus.RESOLVED
    assert resolved.actual_outcome is not None
    assert resolved.actual_outcome.observed_event_id == "event-2"
    assert resolved.prediction_error == pytest.approx(-0.6)
    with pytest.raises(ValueError, match="already recorded"):
        store.record_outcome(
            record.decision_id,
            description="Duplicate",
            utility=0.0,
            success=False,
            observed_event_id=None,
            observed_event_sequence=None,
        )


def test_records_round_trip_through_json() -> None:
    store = DecisionStore()
    store.create(
        [_respond_candidate(), _fallback_candidate()],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id="context-1",
        active_goal_ids=("goal-1",),
        value_revision_refs={"care": 3},
        emotion_snapshot={"valence": 0.2},
        decision_id="decision-1",
    )
    payload = json.loads(json.dumps(store.to_json()))

    restored = DecisionStore()
    restored.restore(payload)

    assert restored.get("decision-1") == store.get("decision-1")


def test_v1_record_migrates_without_self_model_contributions() -> None:
    store = DecisionStore()
    store.create(
        [_fallback_candidate()],
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id="legacy",
    )
    payload = json.loads(json.dumps(store.to_json()))
    payload[0]["schema_version"] = 1
    del payload[0]["adapter_id"]
    del payload[0]["adapter_hash"]
    del payload[0]["activation_sequence"]
    del payload[0]["considered_candidates"][0]["self_model_contributions"]

    restored = DecisionStore()
    restored.restore(payload)

    assert (
        restored.get("legacy").considered_candidates[0].self_model_contributions
        == {}
    )
    assert restored.get("legacy").adapter_id is None


def test_schema_parser_rejects_free_form_reasoning_and_unknown_fields() -> None:
    payload = {"candidates": [_candidate_payload(_fallback_candidate())]}

    parsed = parse_candidate_output(json.dumps(payload))

    assert parsed[0].candidate_type == ActionType.NO_OP
    with pytest.raises(ValueError):
        parse_candidate_output(
            {"candidates": [{**payload["candidates"][0], "hidden_thought": "secret"}]}
        )
    with pytest.raises(ValueError):
        parse_candidate_output("not json")
    prompt = schema_candidate_prompt("Choose safely")
    assert "strict JSON" in prompt
    assert "Do not include reasoning" in prompt


def test_dataset_uses_only_resolved_structured_records_without_hidden_thought() -> None:
    store = DecisionStore()
    unresolved = store.create(
        [_fallback_candidate()],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id="context-1",
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id="unresolved",
    )
    resolved = store.create(
        [_respond_candidate(), _fallback_candidate()],
        triggering_event_id="event-2",
        triggering_event_sequence=2,
        context_id="context-1",
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id="resolved",
    )
    resolved = store.record_outcome(
        resolved.decision_id,
        description="Observed result",
        utility=0.5,
        success=True,
        observed_event_id="event-3",
        observed_event_sequence=3,
    )

    dataset = DecisionDatasetGenerator().generate([unresolved, resolved])

    assert [item.source_id for item in dataset] == ["resolved"]
    serialized = json.dumps(dataset[0].to_json())
    assert "hidden_thought" not in serialized
    assert "reasoning" not in serialized


def test_candidate_schema_validates_bounds_and_requires_fallback() -> None:
    with pytest.raises(ValueError):
        ActionCandidate(
            **{**_respond_candidate().__dict__, "estimated_risk": float("nan")}
        )
    with pytest.raises(ValueError, match="non-action fallback"):
        DecisionStore().create(
            [_respond_candidate()],
            triggering_event_id=None,
            triggering_event_sequence=None,
            context_id=None,
            active_goal_ids=(),
            value_revision_refs={},
            emotion_snapshot={},
        )


def _respond_candidate() -> ActionCandidate:
    return ActionCandidate(
        candidate_id="respond",
        candidate_type=ActionType.RESPOND,
        proposed_action="Provide a concise answer",
        parameters={"style": "concise"},
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome(
                outcome_id="helpful",
                description="The request is satisfied",
                probability=1.0,
                utility=0.8,
            ),
        ),
        uncertainty=0.1,
        estimated_cost=0.1,
        estimated_risk=0.1,
        value_effects={"honesty": 0.5},
        appraisal_contributions={"goal_progress": 0.3},
    )


def _fallback_candidate(
    *, action_type: ActionType = ActionType.NO_OP
) -> ActionCandidate:
    identifier = action_type.value.replace("_", "-")
    return ActionCandidate(
        candidate_id=identifier,
        candidate_type=action_type,
        proposed_action=f"Choose {action_type.value}",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(),
        uncertainty=0.2,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )


def _candidate_payload(candidate: ActionCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type.value,
        "proposed_action": candidate.proposed_action,
        "parameters": candidate.parameters,
        "prerequisites": list(candidate.prerequisites),
        "predicted_outcomes": [
            {
                "outcome_id": item.outcome_id,
                "description": item.description,
                "probability": item.probability,
                "utility": item.utility,
            }
            for item in candidate.predicted_outcomes
        ],
        "uncertainty": candidate.uncertainty,
        "estimated_cost": candidate.estimated_cost,
        "estimated_risk": candidate.estimated_risk,
        "value_effects": candidate.value_effects,
        "appraisal_contributions": candidate.appraisal_contributions,
    }

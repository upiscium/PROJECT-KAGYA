from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json

import pytest

from kagya.cognition import AppraisalResult
from kagya.experience import build_chat_experience
from kagya.identity import OriginActor, OriginInputKind, new_identity_origin
from kagya.motivation import (
    MotivationDynamics,
    MotivationKind,
    MotivationSource,
    MotivationStatus,
)


def test_repeated_experience_forms_bounded_goal_candidate() -> None:
    dynamics = MotivationDynamics(max_goal_proposals_per_cycle=1)

    first = dynamics.observe_experience(_experience("one"))
    assert first
    assert dynamics.goal_candidates()[0] == []

    dynamics.observe_experience(_experience("two"))
    candidates, held = dynamics.goal_candidates(review_at=_after_persistence(first[0]))

    assert len(candidates) == 1
    assert held == ()
    assert candidates[0].target_ref == "context:ctx-interest"
    assert candidates[0].description.startswith("Investigate unresolved novelty")
    record = dynamics.get(candidates[0].motivation_id)
    assert record.evidence_count == 2
    assert record.persistence >= 0.4


def test_duplicate_experience_does_not_reinforce_or_multiply_motivation() -> None:
    dynamics = MotivationDynamics()
    experience = _experience("same")

    dynamics.observe_experience(experience)
    before = dynamics.to_json()
    assert dynamics.observe_experience(experience) == []

    assert dynamics.to_json() == before


def test_conflicting_motivations_are_held_without_deletion() -> None:
    dynamics = MotivationDynamics()
    dynamics.observe_experience(_experience("one"))
    dynamics.observe_experience(_experience("two"))
    dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:planner",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("future-self:planner@1", "identity-claim:planning@1"),
    )
    records = dynamics.list_records()
    assert len(records) >= 2
    left, right = records[:2]
    dynamics.register_conflict(left.motivation_id, right.motivation_id)

    candidates, held = dynamics.goal_candidates(
        review_at=max(_after_persistence(item) for item in records)
    )

    assert candidates == []
    assert set(held) == {left.motivation_id, right.motivation_id}
    assert len(dynamics.list_records()) == len(records)


def test_goal_outcome_and_time_update_drive_state() -> None:
    dynamics = MotivationDynamics()
    dynamics.observe_experience(_experience("one"))
    dynamics.observe_experience(_experience("two"))
    candidate = dynamics.goal_candidates(
        review_at=_after_persistence(dynamics.list_records()[0])
    )[0][0]
    linked = dynamics.link_goal(candidate.motivation_id, "goal-one")

    assert linked.kind.value == "desire"
    satisfied = dynamics.resolve_goal("goal-one", success=True)[0]
    assert satisfied.status == MotivationStatus.SATISFIED
    assert satisfied.satiation == 1.0

    other = MotivationDynamics()
    other.observe_experience(_experience("three"))
    decayed = other.decay(24.0)[0]
    assert decayed.strength < other.list_records()[0].revisions[0].before["strength"]


def test_structured_evidence_is_deduplicated_and_persistent() -> None:
    dynamics = MotivationDynamics(max_goal_proposals_per_cycle=1)

    first = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:planner",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("future-self:planner", "identity-claim:planning"),
    )
    repeated = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:planner",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("future-self:planner", "identity-claim:planning"),
    )

    assert repeated == first
    assert repeated.evidence_count == 2
    assert (
        dynamics.goal_candidates(review_at=_after_persistence(first))[0][
            0
        ].motivation_id
        == first.motivation_id
    )
    episode = dynamics.record_episode(
        selected_ids=(),
        held_ids=(),
        generated_goal_ids=(),
        event_id=None,
        event_sequence=None,
        budget=0,
    )
    assert episode.budget == 0


def test_prediction_error_contributes_to_curiosity_without_novelty() -> None:
    dynamics = MotivationDynamics(min_persistence_seconds=0)
    experience = replace(
        _experience("prediction-error"),
        appraisal=replace(
            _experience("template").appraisal,
            novelty=None,
            novelty_valid=False,
        ),
        prediction_error=2.0,
    )

    records = dynamics.observe_experience(experience)

    assert len(records) == 1
    evidence = records[0].evidence[0]
    measurements = dict(evidence.measurements)
    assert measurements["novelty"] == 0.0
    assert 0.0 < measurements["prediction_error"] < 1.0
    assert 0.0 <= measurements["signal"] <= 1.0


def test_external_request_suggestion_and_constraint_cannot_spoof_intrinsic_evidence() -> (
    None
):
    for kind in (
        OriginInputKind.REQUEST,
        OriginInputKind.SUGGESTION,
        OriginInputKind.CONSTRAINT,
    ):
        dynamics = MotivationDynamics(min_persistence_seconds=0)
        experience = _experience("spoof", input_kind=kind)
        assert dynamics.observe_experience(experience) == []
        assert dynamics.list_records() == []


def test_request_requires_independent_self_origin_curiosity_corroboration() -> None:
    dynamics = MotivationDynamics(min_persistence_seconds=0)
    request = _experience("request", input_kind=OriginInputKind.REQUEST)
    assert dynamics.observe_experience(request) == []

    self_evidence = _experience(
        "self", actor=OriginActor.SELF, input_kind=OriginInputKind.INTERNAL_STATE
    )
    intrinsic = dynamics.observe_experience(self_evidence)[0]
    reinforced = dynamics.observe_experience(
        _experience("corroborated-request", input_kind=OriginInputKind.REQUEST)
    )[0]

    assert reinforced.motivation_id == intrinsic.motivation_id
    assert any(item.origin_actor == OriginActor.SELF for item in reinforced.evidence)


def test_elapsed_persistence_and_saturation_gate_goal_lifecycle() -> None:
    dynamics = MotivationDynamics(min_persistence_seconds=60)
    first = dynamics.observe_experience(_experience("one"))[0]
    dynamics.observe_experience(_experience("two"))

    assert (
        dynamics.goal_candidates(
            review_at=datetime.fromisoformat(first.created_at) + timedelta(seconds=59)
        )[0]
        == []
    )
    candidate = dynamics.goal_candidates(review_at=_after_persistence(first))[0][0]
    dynamics.link_goal(candidate.motivation_id, "goal-one")
    dynamics.resolve_goal("goal-one", success=True)

    unchanged = dynamics.observe_structured_signal(
        MotivationKind.INTEREST,
        MotivationSource.CURIOSITY,
        first.target_ref,
        signal=0.9,
        uncertainty=0.1,
        source_refs=first.source_refs,
    )
    assert unchanged.status == MotivationStatus.SATISFIED
    assert len(dynamics.list_records()) == 1


def test_authoritative_clock_gates_persistence_and_survives_restore() -> None:
    current = datetime(2001, 1, 1, tzinfo=UTC)
    dynamics = MotivationDynamics(
        min_evidence_count=1,
        min_persistence_seconds=60,
        clock=lambda: current,
    )
    record = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:controlled-clock",
        signal=0.9,
        uncertainty=0.1,
        source_refs=("future-self:controlled-clock@1",),
    )

    assert record.created_at == current.isoformat()
    current += timedelta(seconds=59)
    assert dynamics.goal_candidates()[0] == []

    restored = MotivationDynamics(
        min_evidence_count=1,
        min_persistence_seconds=60,
        clock=lambda: current,
    )
    restored.restore(json.loads(json.dumps(dynamics.to_json())))
    assert restored.get(record.motivation_id).created_at == record.created_at
    assert restored.goal_candidates()[0] == []

    current += timedelta(seconds=1)
    assert restored.goal_candidates()[0][0].motivation_id == record.motivation_id


def test_motivation_clock_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MotivationDynamics(clock=lambda: datetime(2001, 1, 1))


def test_terminal_source_reopens_only_for_new_source_revision_and_restores() -> None:
    dynamics = MotivationDynamics(min_evidence_count=1, min_persistence_seconds=0)
    first = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:planner",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("future-self:planner@1",),
    )
    dynamics.link_goal(first.motivation_id, "goal-one")
    dynamics.resolve_goal("goal-one", success=False)

    unchanged = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:planner",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("future-self:planner@1",),
    )
    reopened = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:planner",
        signal=0.9,
        uncertainty=0.1,
        source_refs=("future-self:planner@2",),
    )

    assert unchanged.motivation_id == first.motivation_id
    assert reopened.motivation_id != first.motivation_id
    restored = MotivationDynamics(min_evidence_count=1, min_persistence_seconds=0)
    restored.restore(json.loads(json.dumps(dynamics.to_json())))
    assert restored.get(first.motivation_id).status == MotivationStatus.FAILED
    assert (
        restored.get(reopened.motivation_id).evidence[0].source_state_ref.endswith("@2")
    )


def test_failed_decayed_and_conflicting_motives_survive_restart() -> None:
    dynamics = MotivationDynamics(min_evidence_count=1, min_persistence_seconds=0)
    failed = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "limitation:failed",
        signal=0.9,
        uncertainty=0.1,
        source_refs=("limitation:failed@1",),
    )
    decaying = dynamics.observe_structured_signal(
        MotivationKind.DRIVE,
        MotivationSource.SOCIAL,
        "relationship:decaying",
        signal=0.6,
        uncertainty=0.3,
        source_refs=("relationship:decaying@1",),
    )
    conflicting = dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.DELIBERATION,
        "value:care",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("value:care@1",),
    )
    dynamics.link_goal(failed.motivation_id, "goal-failed")
    dynamics.resolve_goal("goal-failed", success=False)
    dynamics.decay_record(decaying.motivation_id, 20.0)
    dynamics.register_conflict(conflicting.motivation_id, decaying.motivation_id)

    restored = MotivationDynamics(min_evidence_count=1, min_persistence_seconds=0)
    restored.restore(json.loads(json.dumps(dynamics.to_json())))

    assert restored.get(failed.motivation_id).status == MotivationStatus.FAILED
    assert restored.get(decaying.motivation_id).status == MotivationStatus.DECAYED
    assert (
        decaying.motivation_id in restored.get(conflicting.motivation_id).conflict_ids
    )


def _experience(
    identifier: str,
    *,
    actor: OriginActor = OriginActor.USER,
    input_kind: OriginInputKind = OriginInputKind.OBSERVATION,
):
    return build_chat_experience(
        source_event_id=f"event-{identifier}",
        source_event_sequence=1,
        episode_id=f"episode-{identifier}",
        identity_origin=new_identity_origin(
            actor,
            input_kind,
            source_ref="context:ctx-interest",
        ),
        context_id="ctx-interest",
        interlocutor_ids=(),
        appraisal=AppraisalResult(
            novelty=0.9,
            goal_progress=0.0,
            threat=0.0,
            controllability=0.5,
            certainty=0.4,
            social_relevance=0.0,
            effort_cost=0.2,
            novelty_valid=True,
            reasons=("novelty_measured",),
        ),
        valence=0.0,
        arousal=0.5,
        prediction_error=0.8,
        value_revision_refs={"curiosity": 0},
        active_goal_refs=(),
        self_model_revision=0,
    )


def _after_persistence(record) -> datetime:
    return datetime.fromisoformat(record.created_at).astimezone(UTC) + timedelta(
        seconds=61
    )

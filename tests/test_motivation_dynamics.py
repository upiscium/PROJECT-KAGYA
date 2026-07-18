from kagya.cognition import AppraisalResult
from kagya.experience import build_chat_experience
from kagya.identity import OriginActor, OriginInputKind, new_identity_origin
from kagya.motivation import MotivationDynamics, MotivationStatus


def test_repeated_experience_forms_bounded_goal_candidate() -> None:
    dynamics = MotivationDynamics(max_goal_proposals_per_cycle=1)

    first = dynamics.observe_experience(_experience("one"))
    assert first
    assert dynamics.goal_candidates()[0] == []

    dynamics.observe_experience(_experience("two"))
    candidates, held = dynamics.goal_candidates()

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
    records = dynamics.list_records()
    assert len(records) >= 2
    left, right = records[:2]
    dynamics.register_conflict(left.motivation_id, right.motivation_id)

    candidates, held = dynamics.goal_candidates()

    assert candidates == []
    assert set(held) == {left.motivation_id, right.motivation_id}
    assert len(dynamics.list_records()) == len(records)


def test_goal_outcome_and_time_update_drive_state() -> None:
    dynamics = MotivationDynamics()
    dynamics.observe_experience(_experience("one"))
    dynamics.observe_experience(_experience("two"))
    candidate = dynamics.goal_candidates()[0][0]
    linked = dynamics.link_goal(candidate.motivation_id, "goal-one")

    assert linked.kind.value == "desire"
    satisfied = dynamics.resolve_goal("goal-one", success=True)[0]
    assert satisfied.status == MotivationStatus.SATISFIED
    assert satisfied.satiation == 1.0

    other = MotivationDynamics()
    other.observe_experience(_experience("three"))
    decayed = other.decay(24.0)[0]
    assert decayed.strength < other.list_records()[0].revisions[0].before["strength"]


def _experience(identifier: str):
    return build_chat_experience(
        source_event_id=f"event-{identifier}",
        source_event_sequence=1,
        episode_id=f"episode-{identifier}",
        identity_origin=new_identity_origin(
            OriginActor.USER,
            OriginInputKind.OBSERVATION,
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

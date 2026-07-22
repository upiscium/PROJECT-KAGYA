import json

from kagya.cognition import AppraisalResult
from kagya.decision import ActionCandidate, ActionType, DecisionStore, PredictedOutcome
from kagya.experience import ExperienceStore, build_chat_experience
from kagya.identity import (
    IdentityClaimKind,
    IdentityClaimStatus,
    NarrativeSelf,
    OriginActor,
    OriginInputKind,
    new_identity_origin,
)
from kagya.motivation import GoalManager, GoalType, MotivationDynamics


def test_high_importance_experiences_form_episodes_and_chapters() -> None:
    narrative = NarrativeSelf(episode_threshold=0.6)
    first = _experience("one", social_relevance=0.95)
    second = _experience("two", social_relevance=0.9)

    first_episode = narrative.observe_experience(first)
    second_episode = narrative.observe_experience(second)

    assert first_episode is not None
    assert second_episode is not None
    assert first_episode.experience_ids == (first.experience_id,)
    assert len(narrative.chapters) == 1
    chapter = next(iter(narrative.chapters.values()))
    assert chapter.episode_ids == (first_episode.episode_id, second_episode.episode_id)


def test_low_importance_experience_is_not_autobiographical() -> None:
    narrative = NarrativeSelf(episode_threshold=0.6)

    assert narrative.observe_experience(_experience("ordinary")) is None
    assert narrative.episodes == {}


def test_trait_requires_repeated_episodes_and_conflicting_claims_coexist() -> None:
    narrative = NarrativeSelf()
    first = _experience("one", social_relevance=0.95)
    second = _experience("two", social_relevance=0.95)
    narrative.observe_experience(first)
    narrative.observe_experience(second)

    provisional = narrative.propose_claim(
        kind=IdentityClaimKind.TRAIT,
        statement="I persist through difficult work",
        polarity=1,
        theme_codes=("persistence",),
        confidence=0.9,
        stability=0.7,
        evidence_refs=(f"experience:{first.experience_id}",),
        related_experience_ids=(first.experience_id,),
        claim_id="persistent",
    )
    supported = narrative.propose_claim(
        kind=IdentityClaimKind.TRAIT,
        statement="I sometimes withdraw from difficult work",
        polarity=-1,
        theme_codes=("persistence",),
        confidence=0.7,
        stability=0.5,
        evidence_refs=(
            f"experience:{first.experience_id}",
            f"experience:{second.experience_id}",
        ),
        related_experience_ids=(first.experience_id, second.experience_id),
        claim_id="withdrawing",
    )

    assert provisional.confidence == 0.49
    assert provisional.status == IdentityClaimStatus.HYPOTHESIS
    assert supported.status == IdentityClaimStatus.CONTESTED
    assert narrative.get_claim("persistent").status == IdentityClaimStatus.CONTESTED
    assert set(narrative.claims) == {"persistent", "withdrawing"}
    assert len(narrative.conflicts) == 1


def test_counterevidence_and_round_trip_preserve_revision_history() -> None:
    narrative = NarrativeSelf()
    experience = _experience("one", social_relevance=0.95)
    narrative.observe_experience(experience)
    narrative.propose_claim(
        kind=IdentityClaimKind.IDENTITY,
        statement="I am reliable",
        polarity=1,
        theme_codes=("reliability",),
        confidence=0.8,
        stability=0.8,
        evidence_refs=(f"experience:{experience.experience_id}",),
        related_experience_ids=(experience.experience_id,),
        claim_id="reliable",
    )

    revised = narrative.revise_claim(
        "reliable",
        confidence=0.55,
        reason_code="missed_commitment",
        counterevidence_refs=("decision:failed",),
    )
    commitment_event = narrative.record_commitment_event(
        "commitment:one",
        kind="breach",
        description="A responsibility was not fulfilled",
        evidence_refs=("decision:failed",),
        relationship_refs=("relationship:one",),
    )
    restored = NarrativeSelf()
    restored.restore(json.loads(json.dumps(narrative.to_json())))

    assert revised.status == IdentityClaimStatus.CONTESTED
    assert revised.counterevidence_refs == ("decision:failed",)
    assert restored.get_claim("reliable") == revised
    assert restored.commitment_events[commitment_event.event_id] == commitment_event


def test_value_change_and_failure_are_preserved_as_turning_points() -> None:
    narrative = NarrativeSelf()
    experience_store = ExperienceStore()
    original = experience_store.integrate(
        _experience("failure", social_relevance=0.95, goal_progress=-0.8)
    )
    episode = narrative.observe_experience(original)
    assert episode is not None
    linked = experience_store.link_result(
        original.experience_id,
        kind="value",
        reference="value:care@2",
        evidence_refs=(f"experience:{original.experience_id}",),
    )

    refreshed = narrative.observe_experience(linked)

    assert refreshed is not None
    assert refreshed.turning_point is True
    assert set(refreshed.turning_point_codes) == {
        "significant_failure",
        "value_change",
    }
    assert refreshed.related_value_refs[-1] == "value:care@2"


def test_future_self_gap_creates_persistent_motivation() -> None:
    narrative = NarrativeSelf()
    dynamics = MotivationDynamics()
    projection = narrative.set_future_self(
        description="Become more capable at planning",
        theme_codes=("planning",),
        desired_level=0.9,
        current_level=0.3,
        evidence_refs=("identity-claim:planning",),
        projection_id="planner",
    )

    motivation = dynamics.observe_future_self_gap(
        projection.projection_id, gap=projection.gap, uncertainty=0.2
    )
    assert motivation is not None
    narrative.link_future_motivation("planner", motivation.motivation_id)

    assert motivation.target_ref == "future-self:planner"
    assert narrative.future_self["planner"].related_motivation_ids == (
        motivation.motivation_id,
    )
    restored = MotivationDynamics()
    restored.restore(json.loads(json.dumps(dynamics.to_json())))
    assert restored.get(motivation.motivation_id) == motivation


def test_goal_and_decision_persist_narrative_provenance() -> None:
    goals = GoalManager()
    goal = goals.propose(
        goal_type=GoalType.INTRINSIC,
        description="Practice planning",
        narrative_self_refs=("identity-claim:planner",),
        goal_id="planning-goal",
    )
    restored_goals = GoalManager()
    restored_goals.restore(json.loads(json.dumps(goals.goals_json())))

    decisions = DecisionStore()
    decision = decisions.create(
        [_candidate()],
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(goal.goal_id,),
        value_revision_refs={},
        emotion_snapshot={},
        narrative_self_refs=("identity-claim:planner",),
        decision_id="decision-one",
    )
    restored_decisions = DecisionStore()
    restored_decisions.restore(json.loads(json.dumps(decisions.to_json())))

    assert restored_goals.get(goal.goal_id).narrative_self_refs == (
        "identity-claim:planner",
    )
    assert decision.narrative_self_refs == ("identity-claim:planner",)
    assert restored_decisions.get(decision.decision_id) == decision


def _experience(
    identifier: str,
    *,
    social_relevance: float = 0.0,
    goal_progress: float = 0.0,
):
    return build_chat_experience(
        source_event_id=f"event-{identifier}",
        source_event_sequence=1,
        episode_id=f"memory-{identifier}",
        identity_origin=new_identity_origin(
            OriginActor.USER,
            OriginInputKind.OBSERVATION,
            source_ref="context:narrative",
        ),
        context_id="narrative-context",
        interlocutor_ids=(),
        appraisal=AppraisalResult(
            novelty=0.1,
            goal_progress=goal_progress,
            threat=0.0,
            controllability=0.6,
            certainty=0.8,
            social_relevance=social_relevance,
            effort_cost=0.1,
            novelty_valid=True,
            reasons=("shared_theme",),
        ),
        valence=-0.5 if goal_progress < 0 else 0.2,
        arousal=0.4,
        prediction_error=0.2,
        value_revision_refs={"care": 1},
        active_goal_refs=(),
        self_model_revision=0,
    )


def _candidate() -> ActionCandidate:
    return ActionCandidate(
        candidate_id="observe",
        candidate_type=ActionType.OBSERVE,
        proposed_action="Observe",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome(
                outcome_id="outcome",
                description="Learn",
                probability=1.0,
                utility=0.1,
            ),
        ),
        uncertainty=0.0,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )

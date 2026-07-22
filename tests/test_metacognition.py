import json

from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionStore,
    PredictedOutcome,
)
from kagya.identity import (
    Capability,
    CapabilityEvidence,
    KnownLimitation,
    SelfModelState,
)
from kagya.metacognition import EpistemicBoundary, Metacognition


def test_confidence_tracks_evidence_quantity_and_past_accuracy() -> None:
    metacognition = Metacognition()
    sparse = _state(evidence=(_evidence("one", True),))
    strong = _state(
        evidence=tuple(_evidence(str(index), True) for index in range(8))
    )

    sparse_assessment = _assess(metacognition, "sparse", sparse)
    strong_assessment = _assess(metacognition, "strong", strong)
    resolved = _resolved_decision(
        "strong",
        pre_assessment_id=strong_assessment.assessment_id,
        success=False,
        utility=-0.8,
    )
    metacognition.assess_post(
        resolved,
        self_model_revision=strong.revision,
        cognitive_quality=strong_assessment.cognitive_quality,
    )
    recalibrated = _assess(metacognition, "after-failure", strong)

    assert strong_assessment.calibrated_confidence > sparse_assessment.calibrated_confidence
    assert recalibrated.calibrated_confidence < strong_assessment.calibrated_confidence
    assert recalibrated.historical_accuracy is not None


def test_boundaries_choose_structured_information_defer_and_delegate_actions() -> None:
    unknown = _assess(Metacognition(), "unknown", _state(evidence=()))
    unable_state = _state(
        evidence=(_evidence("failure", False),),
        limitation=KnownLimitation(
            limitation_id="cannot-code",
            description="Execution unavailable",
            confidence=1.0,
            capability_ids=("coding",),
            tags=("software",),
            evidence_refs=("system:tool-unavailable",),
        ),
    )
    unable = _assess(Metacognition(), "unable", unable_state)

    assert unknown.boundary == EpistemicBoundary.UNKNOWN
    assert unknown.recommended_action == ActionType.REQUEST_INFORMATION
    assert unable.boundary == EpistemicBoundary.UNABLE
    assert unable.recommended_action == ActionType.DELEGATE


def test_quality_detects_load_attention_and_emotion_without_self_report() -> None:
    metacognition = Metacognition()
    assessment = metacognition.assess_pre(
        "quality",
        _candidates(),
        self_model=_state(evidence=(_evidence("one", True),)),
        narrative_self_refs=("identity-claim:coding@2",),
        cognitive_load=0.9,
        attention_saturation=1.0,
        emotion_valence=-0.8,
        emotion_arousal=0.9,
        quality_provenance_refs=("working-memory:occupancy:9", "attention-focus@3", "emotion:current"),
    )

    assert assessment.cognitive_quality.estimated_quality < 0.2
    assert set(assessment.cognitive_quality.reason_codes) == {
        "high_cognitive_load",
        "attention_saturated",
        "emotion_may_degrade_judgment",
    }
    serialized = json.dumps(metacognition.to_json())
    assert "hidden_thought" not in serialized
    assert "apology" not in serialized


def test_prediction_outcomes_form_revisioned_recurring_error_hypothesis() -> None:
    metacognition = Metacognition()
    state = _state(evidence=tuple(_evidence(str(index), True) for index in range(5)))
    for index in range(2):
        decision_id = f"failure-{index}"
        pre = _assess(metacognition, decision_id, state)
        post = metacognition.assess_post(
            _resolved_decision(
                decision_id,
                pre_assessment_id=pre.assessment_id,
                success=False,
                utility=-0.8,
            ),
            self_model_revision=state.revision,
            cognitive_quality=pre.cognitive_quality,
        )

    assert post.hypothesis_refs
    hypothesis = metacognition.hypotheses[post.hypothesis_refs[0]]
    assert hypothesis.hypothesis_code in {"recurring_failure", "optimism_bias"}
    assert len(hypothesis.evidence_refs) == 2
    metacognition.withdraw_outcome("failure-1")
    assert metacognition.hypotheses == {}
    restored = Metacognition()
    restored.restore(json.loads(json.dumps(metacognition.to_json())))
    assert restored.to_json() == metacognition.to_json()


def test_operator_feedback_is_explicit_calibration_provenance() -> None:
    metacognition = Metacognition()
    state = _state(evidence=(_evidence("one", True),))
    pre = _assess(metacognition, "feedback", state)
    store = DecisionStore()
    store.create(
        _candidates(),
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id="feedback",
        metacognition_pre_assessment_id=pre.assessment_id,
    )
    decision = store.record_feedback_outcome(
        "feedback",
        utility=-0.75,
        success=False,
        feedback_id="operator-1",
        feedback_revision=2,
        observed_event_id="event-1",
        observed_event_sequence=1,
    )

    post = metacognition.assess_post(
        decision,
        self_model_revision=state.revision,
        cognitive_quality=pre.cognitive_quality,
    )

    assert "feedback:operator-1@2" in post.evidence_refs
    assert "operator_feedback_calibration" in post.reason_codes


def _assess(
    metacognition: Metacognition, decision_id: str, state: SelfModelState
):
    return metacognition.assess_pre(
        decision_id,
        _candidates(),
        self_model=state,
        narrative_self_refs=("identity-claim:coding@2",),
        cognitive_load=0.1,
        attention_saturation=0.1,
        emotion_valence=0.0,
        emotion_arousal=0.1,
        quality_provenance_refs=("working-memory:occupancy:1", "attention-focus@1", "emotion:current"),
    )


def _state(
    *,
    evidence: tuple[CapabilityEvidence, ...],
    limitation: KnownLimitation | None = None,
) -> SelfModelState:
    capabilities = (
        {}
        if not evidence
        else {
            "coding": Capability(
                capability_id="coding",
                description="Implement software",
                confidence=0.8,
                stability=0.7,
                tags=("software",),
                evidence=evidence,
            )
        }
    )
    return SelfModelState(
        identity_summary="Test subject",
        traits={},
        capabilities=capabilities,
        known_limitations={} if limitation is None else {limitation.limitation_id: limitation},
        epistemic_uncertainties={},
        roles=(),
        commitment_refs=(),
        autobiographical_summary_refs=(),
        revision=2,
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _evidence(identifier: str, success: bool) -> CapabilityEvidence:
    return CapabilityEvidence(
        evidence_id=f"decision:{identifier}",
        evidence_type="decision_outcome",
        source_id=identifier,
        success=success,
        utility=1.0 if success else -1.0,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _candidates() -> tuple[ActionCandidate, ...]:
    return (
        _candidate("respond", ActionType.RESPOND, probability=0.9, utility=0.8),
        _candidate("request", ActionType.REQUEST_INFORMATION),
        _candidate("defer", ActionType.DEFER),
        _candidate("delegate", ActionType.DELEGATE),
    )


def _candidate(
    identifier: str,
    action_type: ActionType,
    *,
    probability: float = 0.5,
    utility: float = 0.0,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=identifier,
        candidate_type=action_type,
        proposed_action=identifier,
        parameters={"capability_ids": ["coding"], "topic_tags": ["software"]},
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome("result", "Structured result", probability, utility),
        ),
        uncertainty=1.0 - probability,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )


def _resolved_decision(
    decision_id: str,
    *,
    pre_assessment_id: str,
    success: bool,
    utility: float,
):
    store = DecisionStore()
    candidates = _candidates()
    store.create(
        candidates,
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id=decision_id,
        metacognition_pre_assessment_id=pre_assessment_id,
    )
    decision = store.record_outcome(
        decision_id,
        description="Observed structured result",
        utility=utility,
        success=success,
        observed_event_id=f"event:{decision_id}",
        observed_event_sequence=1,
    )
    return decision

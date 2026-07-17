import json

import pytest

from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionStore,
    PredictedOutcome,
)
from kagya.identity import (
    EpistemicUncertainty,
    KnownLimitation,
    ProposalStatus,
    SelfModel,
)


def test_resolved_success_and_failure_update_capability_with_evidence() -> None:
    model = SelfModel(max_capability_update=0.1)
    success = _resolved_decision("success", success=True, utility=0.8)
    failure = _resolved_decision("failure", success=False, utility=-0.7)

    after_success = model.update_capability_from_decision(
        "writing", "Write useful answers", success, tags=("text",)
    )
    after_failure = model.update_capability_from_decision(
        "writing", "Write useful answers", failure, tags=("text",)
    )

    assert 0.5 < after_success.confidence <= 0.53
    assert after_failure.confidence < after_success.confidence
    assert [item.source_id for item in after_failure.evidence] == [
        "success",
        "failure",
    ]
    assert model.history[-1].evidence_refs == ("decision:failure",)


def test_unresolved_decision_and_self_report_cannot_confirm_capability() -> None:
    model = SelfModel()
    unresolved = _unresolved_decision("unresolved")

    with pytest.raises(ValueError, match="resolved DecisionRecord"):
        model.update_capability_from_decision(
            "coding", "Write code", unresolved
        )
    proposal = model.propose_identity_revision(
        proposed_summary=None,
        proposed_traits={"capable": 1.0},
        evidence_refs=(),
        source="self_report",
        proposal_id="claim",
    )

    assert proposal.status == ProposalStatus.PENDING
    assert model.state.capabilities == {}
    assert model.state.traits == {}


def test_unrelated_decision_cannot_update_capability() -> None:
    model = SelfModel()
    decision = _resolved_decision("unrelated", success=True, utility=1.0)

    with pytest.raises(ValueError, match="did not declare"):
        model.update_capability_from_decision(
            "coding", "Write code", decision
        )


def test_limitation_and_uncertainty_penalize_relevant_candidate_only() -> None:
    model = SelfModel()
    model.manual_correct_capability(
        "coding",
        "Write code",
        0.8,
        reason="verified correction",
        tags=("software",),
    )
    model.add_limitation(
        KnownLimitation(
            limitation_id="no-network",
            description="Cannot access public networks",
            confidence=1.0,
            capability_ids=("coding",),
            tags=("network",),
            evidence_refs=("manual:network",),
        ),
        reason="deployment boundary",
    )
    model.add_uncertainty(
        EpistemicUncertainty(
            uncertainty_id="unknown-api",
            description="API behavior is unknown",
            confidence=0.8,
            tags=("network",),
            evidence_refs=("observation:missing",),
        ),
        reason="missing evidence",
    )
    relevant = _candidate(
        "network-action",
        ActionType.INTERNAL,
        parameters={
            "capability_ids": ["coding"],
            "topic_tags": ["network"],
        },
    )
    unrelated = _candidate(
        "local-action",
        ActionType.INTERNAL,
        parameters={"topic_tags": ["local"]},
    )

    relevant_selection = model.select_relevant(relevant)
    unrelated_selection = model.select_relevant(unrelated)

    assert relevant_selection.capability_ids == ("coding",)
    assert relevant_selection.limitation_ids == ("no-network",)
    assert relevant_selection.uncertainty_ids == ("unknown-api",)
    assert sum(relevant_selection.contributions.values()) < 0
    assert unrelated_selection.rendered_items == ()


def test_self_model_contributions_change_decision_scores() -> None:
    model = SelfModel()
    model.add_limitation(
        KnownLimitation(
            limitation_id="unsafe",
            description="Unsafe remote operation",
            confidence=1.0,
            capability_ids=(),
            tags=("remote",),
            evidence_refs=("manual",),
        ),
        reason="safety boundary",
    )
    risky = _candidate(
        "risky",
        ActionType.INTERNAL,
        parameters={"topic_tags": ["remote"]},
        utility=0.4,
    )
    fallback = _candidate("defer", ActionType.DEFER, utility=0.0)

    record = DecisionStore().create(
        [risky, fallback],
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        self_model_evaluator=model.evaluate_candidates,
    )

    assert record.selected_candidate_id == "defer"
    risky_evaluation = record.considered_candidates[0]
    assert risky_evaluation.self_model_contributions == {"limitation:unsafe": -0.5}


def test_identity_conflict_stays_pending_then_applies_with_inertia() -> None:
    model = SelfModel(max_trait_update=0.1)
    first = model.propose_identity_revision(
        proposed_summary="A different identity",
        proposed_traits={"cautious": 1.0},
        evidence_refs=("decision:one",),
        source="reflection",
        proposal_id="proposal-1",
    )

    assert first.status == ProposalStatus.PENDING
    assert "identity_summary_changed" in first.contradictions
    assert model.state.identity_summary != "A different identity"
    resolved = model.resolve_identity_revision(
        "proposal-1", apply=True, reason="admin reviewed"
    )

    assert resolved.status == ProposalStatus.APPLIED
    assert resolved.identity_origin.endorsement.value == "endorsed"
    assert model.state.identity_summary == "A different identity"
    assert model.state.traits["cautious"] == pytest.approx(0.1)
    assert model.history[-1].evidence_refs == ("decision:one",)

    second = model.propose_identity_revision(
        proposed_summary=None,
        proposed_traits={"cautious": -1.0},
        evidence_refs=("decision:two",),
        source="reflection",
        proposal_id="proposal-2",
    )
    assert "trait_change_exceeds_inertia:cautious" in second.contradictions
    assert "trait_direction_conflict:cautious" in second.contradictions


def test_rollback_and_json_round_trip_preserve_history() -> None:
    model = SelfModel()
    model.manual_correct_capability(
        "writing", "Write text", 0.9, reason="manual verification"
    )
    assert model.state.revision == 1

    rolled_back = model.rollback(0, reason="undo correction")

    assert rolled_back.capabilities == {}
    assert rolled_back.revision == 2
    assert model.history[-1].rollback_target_revision == 0
    payload = json.loads(json.dumps(model.to_json()))
    restored = SelfModel()
    restored.restore(payload)
    assert restored.state == model.state
    assert len(restored.history) == len(model.history)


def test_empty_snapshot_resets_self_model_state_and_history() -> None:
    model = SelfModel()
    model.manual_correct_capability(
        "writing", "Write text", 0.9, reason="manual verification"
    )

    model.restore({})

    assert model.state.capabilities == {}
    assert model.state.revision == 0
    assert model.history == []


def test_legacy_identity_proposal_migrates_as_inherited_and_uncertain() -> None:
    model = SelfModel()
    model.propose_identity_revision(
        proposed_summary=None,
        proposed_traits={"careful": 0.5},
        evidence_refs=(),
        source="legacy",
        proposal_id="legacy-proposal",
    )
    payload = model.to_json()
    del payload["proposals"][0]["identity_origin"]

    restored = SelfModel()
    restored.restore(payload)

    origin = restored.proposals["legacy-proposal"].identity_origin
    assert origin.actor.value == "inherited"
    assert origin.endorsement.value == "uncertain"


def _resolved_decision(
    decision_id: str,
    *,
    success: bool,
    utility: float,
    capability_id: str = "writing",
):
    store = DecisionStore()
    store.create(
        [
            _candidate(
                "no-op",
                ActionType.NO_OP,
                parameters={"capability_ids": [capability_id]},
            )
        ],
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id=decision_id,
    )
    return store.record_outcome(
        decision_id,
        description="Observed outcome",
        utility=utility,
        success=success,
        observed_event_id=f"event:{decision_id}",
        observed_event_sequence=1,
    )


def _unresolved_decision(decision_id: str):
    return DecisionStore().create(
        [_candidate("no-op", ActionType.NO_OP)],
        triggering_event_id=None,
        triggering_event_sequence=None,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
        decision_id=decision_id,
    )


def _candidate(
    candidate_id: str,
    action_type: ActionType,
    *,
    parameters: dict[str, object] | None = None,
    utility: float = 0.0,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        candidate_type=action_type,
        proposed_action=candidate_id,
        parameters=parameters or {},
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome(
                outcome_id=f"outcome:{candidate_id}",
                description="Predicted outcome",
                probability=1.0,
                utility=utility,
            ),
        ),
        uncertainty=0.0,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )

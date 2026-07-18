from dataclasses import replace
import json

import pytest

from kagya.cognition import (
    AppraisalResult,
    EvidenceDirection,
    ValueConflictDefinition,
    ValueEvidence,
    ValueState,
    ValueScope,
    ValueSystem,
    ValueUpdateKind,
)
from kagya.identity import OriginActor, OriginInputKind, new_identity_origin


def test_updates_are_bounded_stable_and_idempotent() -> None:
    system = _system()
    proposals = system.proposals_from_appraisal(
        _appraisal(),
        {"care": 1.0, "honesty": -1.0},
        kind=ValueUpdateKind.OUTCOME,
        evidence=ValueEvidence(event_id="event-1", event_sequence=1),
        proposal_id="outcome-1",
    )

    first = system.apply(proposals)
    second = system.apply(proposals)

    assert len(first) == 2
    assert second == []
    assert sum(abs(record.applied_delta) for record in first) <= 0.1
    assert all(abs(record.applied_delta) <= 0.05 for record in first)
    assert system.get("care").weight > 0.6
    assert system.get("honesty").weight < 0.6


def test_freeze_rejects_updates_and_rollback_reset_restore_state() -> None:
    system = _system()
    system.freeze("care", frozen=True)
    proposal = system.proposals_from_appraisal(
        _appraisal(),
        {"care": 1.0},
        kind=ValueUpdateKind.REFLECTION,
        evidence=ValueEvidence(memory_ids=("memory-1",)),
        proposal_id="reflection-1",
    )[0]

    rejected = system.apply([proposal])[0]

    assert rejected.operation == "rejected"
    assert system.get("care").weight == 0.6
    assert system.get("care").revision == 1
    system.freeze("care", frozen=False)
    system.apply([replace(proposal, proposal_id="reflection-2")])
    assert system.get("care").weight > 0.6
    assert system.rollback("care", target_revision=1).weight == 0.6
    system.apply([replace(proposal, proposal_id="reflection-3")])
    assert system.reset(("care",))[0].weight == 0.6


def test_evaluation_exposes_per_value_contributions_and_conflicts() -> None:
    system = _system()

    scores = system.evaluate(
        {
            "gentle_truth": {"care": 0.7, "honesty": 0.8},
            "blunt_truth": {"care": -0.5, "honesty": 1.0},
        }
    )

    assert scores[0].option_id == "gentle_truth"
    assert {item.value_id for item in scores[0].contributions} == {"care", "honesty"}
    assert scores[1].conflicts == ("compassionate-honesty",)


def test_different_value_states_change_option_ranking() -> None:
    care_first = _system()
    care_first.restore({"care": 0.9, "honesty": 0.1})
    honesty_first = _system()
    honesty_first.restore({"care": 0.1, "honesty": 0.9})
    options = {
        "careful": {"care": 1.0},
        "candid": {"honesty": 1.0},
    }

    assert (
        care_first.evaluate(options)[0].total_score
        > care_first.evaluate(options)[1].total_score
    )
    assert (
        honesty_first.evaluate(options)[0].total_score
        < honesty_first.evaluate(options)[1].total_score
    )


def test_legacy_flat_values_migrate_and_round_trip() -> None:
    system = _system()
    system.restore({"care": 0.9, "curiosity": 0.7})

    payload = system.to_json()
    restored = _system()
    restored.restore(payload)

    assert restored.get("care").weight == 0.9
    assert restored.get("curiosity").weight == 0.7
    assert restored.to_json() == payload


def test_json_snapshot_preserves_update_evidence_and_history() -> None:
    system = _system()
    proposal = system.proposals_from_appraisal(
        _appraisal(),
        {"care": 0.5},
        kind=ValueUpdateKind.OBSERVATION,
        evidence=ValueEvidence(
            event_id="event-2",
            event_sequence=2,
            memory_ids=("memory-2",),
            source="test.observation",
        ),
        proposal_id="observation-2",
    )[0]
    system.apply([proposal])
    serialized = json.loads(json.dumps(system.to_json()))

    restored = _system()
    restored.restore(serialized)

    assert restored.history[0].event_id == "event-2"
    assert restored.history[0].memory_ids == ("memory-2",)
    assert restored.history[0].before["weight"] == 0.6
    assert restored.history[0].after["weight"] > 0.6
    assert restored.get("care").schema_version == 3
    assert restored.history[0].identity_origin.actor.value == "inherited"
    assert json.loads(json.dumps(restored.to_json())) == serialized


def test_invalid_value_references_are_rejected() -> None:
    system = _system()

    with pytest.raises(ValueError):
        system.evaluate({"option": {"missing": 1.0}})


def test_external_request_is_evidence_without_directly_updating_strength() -> None:
    system = _system()
    proposal = system.proposals_from_appraisal(
        _appraisal(),
        {"care": 1.0},
        kind=ValueUpdateKind.OBSERVATION,
        evidence=ValueEvidence(
            identity_origin=new_identity_origin(
                OriginActor.USER,
                OriginInputKind.REQUEST,
                source_ref="context:request",
            )
        ),
        proposal_id="external-request",
    )[0]

    record = system.apply([proposal])[0]

    assert record.operation == "evidence_only"
    assert record.applied_delta == 0.0
    assert system.get("care").weight == 0.6
    assert system.get("care").supporting_evidence_ids == (
        "proposal:external-request:care",
    )
    assert system.evidence["proposal:external-request:care"].identity_origin.actor == (
        OriginActor.USER
    )


def test_repeated_independent_evidence_reinforces_with_durable_provenance() -> None:
    system = _system()

    first = system.apply(
        system.proposals_from_appraisal(
            _appraisal(),
            {"care": 1.0},
            kind=ValueUpdateKind.OBSERVATION,
            evidence=ValueEvidence(
                evidence_id="evidence-1",
                experience_ids=("experience-1",),
                context_id="context-a",
            ),
            proposal_id="reinforce-1",
        )
    )[0]
    second = system.apply(
        system.proposals_from_appraisal(
            _appraisal(),
            {"care": 1.0},
            kind=ValueUpdateKind.OBSERVATION,
            evidence=ValueEvidence(
                evidence_id="evidence-2",
                experience_ids=("experience-2",),
                context_id="context-a",
            ),
            proposal_id="reinforce-2",
        )
    )[0]

    state = system.get("care")
    assert second.applied_delta > first.applied_delta > 0
    assert state.origin_experience_ids == ("experience-1", "experience-2")
    assert state.supporting_evidence_ids == ("evidence-1", "evidence-2")
    assert system.evidence["evidence-1"].direction == EvidenceDirection.SUPPORTING
    assert second.revision_diff is not None
    assert "weight" in second.revision_diff.changed_fields

    replay = system.apply(
        system.proposals_from_appraisal(
            _appraisal(),
            {"care": 1.0},
            kind=ValueUpdateKind.OBSERVATION,
            evidence=ValueEvidence(evidence_id="evidence-2"),
            proposal_id="replayed-under-new-proposal",
        )
    )
    assert replay == []
    assert system.get("care") == state


def test_context_values_do_not_leak_and_protected_values_resist_one_event() -> None:
    scoped = replace(
        _system().get("care"),
        scope=ValueScope.CONTEXT,
        context_ids=("context-a",),
        protectedness=1.0,
        negotiability=0.2,
    )
    system = ValueSystem(seeds=[scoped], max_update_per_event=0.05)

    assert (
        system.evaluate({"option": {"care": 1.0}}, context_id="context-b")[
            0
        ].contributions
        == ()
    )
    assert system.evaluate({"option": {"care": 1.0}}, context_id="context-a")[
        0
    ].contributions
    before = system.get("care")
    update = system.apply(
        system.proposals_from_appraisal(
            _appraisal(),
            {"care": -1.0},
            kind=ValueUpdateKind.OUTCOME,
            evidence=ValueEvidence(evidence_id="single-challenge"),
            proposal_id="single-challenge",
        )
    )[0]
    after = system.get("care")

    assert abs(update.applied_delta) < 0.001
    assert after.polarity == before.polarity
    assert after.opposing_evidence_ids == ("single-challenge",)


def test_conflicting_values_are_retained_in_tradeoff_records() -> None:
    system = _system()
    scores = system.evaluate({"blunt": {"care": -1.0, "honesty": 1.0}})

    records = system.record_tradeoffs(
        scores, context_id="context-a", decision_id="decision-1"
    )

    assert records[0].conflict_names == ("compassionate-honesty",)
    assert records[0].value_revision_refs == {"care": 0, "honesty": 0}
    assert records[0].reasoning_codes == (
        "simultaneous_values_retained",
        "tradeoff_required",
    )
    assert set(system.values) == {"care", "honesty"}


def test_polarity_reversal_requires_repeated_opposing_evidence() -> None:
    seed = replace(
        _system().get("care"),
        weight=0.01,
        stability=0.0,
        protectedness=0.0,
    )
    system = ValueSystem(seeds=[seed], max_update_per_event=0.05)

    for index in range(2):
        system.apply(
            system.proposals_from_appraisal(
                _appraisal(),
                {"care": -1.0},
                kind=ValueUpdateKind.OUTCOME,
                evidence=ValueEvidence(evidence_id=f"challenge-{index}"),
                proposal_id=f"challenge-{index}",
            )
        )

    assert system.get("care").polarity == 1
    assert system.get("care").weight == 0.0
    system.apply(
        system.proposals_from_appraisal(
            _appraisal(),
            {"care": -1.0},
            kind=ValueUpdateKind.OUTCOME,
            evidence=ValueEvidence(evidence_id="challenge-2"),
            proposal_id="challenge-2",
        )
    )
    assert system.get("care").polarity == -1
    assert system.get("care").weight > 0.0


def _system() -> ValueSystem:
    return ValueSystem(
        seeds=[
            ValueState(
                value_id="care",
                name="Care",
                weight=0.6,
                confidence=1.0,
                stability=0.5,
                source="test",
                origin="test",
                last_updated_at="2026-01-01T00:00:00+00:00",
                allowed_update_rate=0.05,
            ),
            ValueState(
                value_id="honesty",
                name="Honesty",
                weight=0.6,
                confidence=1.0,
                stability=0.5,
                source="test",
                origin="test",
                last_updated_at="2026-01-01T00:00:00+00:00",
                allowed_update_rate=0.05,
            ),
        ],
        conflicts=[ValueConflictDefinition("care", "honesty", "compassionate-honesty")],
        max_update_per_event=0.05,
        max_total_update_per_event=0.1,
    )


def _appraisal() -> AppraisalResult:
    return AppraisalResult(
        novelty=0.5,
        goal_progress=1.0,
        threat=0.0,
        controllability=1.0,
        certainty=1.0,
        social_relevance=1.0,
        effort_cost=0.0,
        novelty_valid=True,
        reasons=("test",),
    )

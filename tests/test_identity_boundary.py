from dataclasses import replace

import pytest

from kagya.belief import (
    BeliefEvidence,
    BeliefLifecycle,
    BeliefStore,
    EpistemicStatus,
    Proposition,
)
from kagya.cognition import (
    ValueEvidence,
    ValueState,
    ValueSystem,
    ValueUpdateKind,
    ValueUpdateProposal,
)
from kagya.identity import (
    BoundaryAssessmentInput,
    BoundaryClassification,
    BoundaryRecommendation,
    EndorsementStatus,
    IdentityBoundaryStore,
    OriginActor,
    OriginInputKind,
    SocialPressureMetadata,
    SocialPressureSignalType,
    new_identity_origin,
)


def test_aligned_care_requires_self_authority_and_other_welfare_evidence() -> None:
    store = IdentityBoundaryStore()

    care = store.assess(
        BoundaryAssessmentInput(
            action_ref="action:help",
            origin_refs=("origin:self",),
            evidence_refs=("observation:need",),
            self_endorsed_value_refs=("care",),
            other_welfare_evidence_refs=("observation:welfare",),
        ),
        event_id="event-1",
        event_sequence=1,
        value_revision_refs={"care": 3},
        goal_revision_refs={},
        commitment_revision_refs={},
        relationship_revision_refs={"relationship:one": 2},
    )

    assert care.classification == BoundaryClassification.CARE
    assert care.recommendation == BoundaryRecommendation.CARE
    assert care.value_revision_refs == {"care": 3}


def test_pressure_only_and_protected_conflict_never_authorize_action() -> None:
    store = IdentityBoundaryStore()
    signal = store.add_pressure(
        SocialPressureMetadata(
            signal_type=SocialPressureSignalType.CLAIMED_AUTHORITY,
            authority_ref="authority:claim-1",
        ),
        context_id="context:one",
        event_id="event-1",
        event_sequence=1,
    )
    pressure_only = store.assess(
        BoundaryAssessmentInput(
            action_ref="action:comply",
            origin_refs=("origin:user",),
            pressure_signal_ids=(signal.signal_id,),
        ),
        event_id="event-2",
        event_sequence=2,
        value_revision_refs={},
        goal_revision_refs={},
        commitment_revision_refs={},
        relationship_revision_refs={},
    )
    conflict = store.assess(
        BoundaryAssessmentInput(
            action_ref="action:help",
            origin_refs=("origin:self",),
            self_endorsed_value_refs=("care",),
            other_welfare_evidence_refs=("observation:welfare",),
            protected_state_conflict_refs=("commitment:protected@4",),
        ),
        event_id="event-3",
        event_sequence=3,
        value_revision_refs={"care": 2},
        goal_revision_refs={},
        commitment_revision_refs={"protected": 4},
        relationship_revision_refs={},
    )

    assert pressure_only.classification == BoundaryClassification.APPEASEMENT_RISK
    assert pressure_only.recommendation == BoundaryRecommendation.DEFER
    assert conflict.classification == BoundaryClassification.APPEASEMENT_RISK
    assert conflict.recommendation == BoundaryRecommendation.DEFER


def test_chat_repetition_only_creates_repeated_request_fingerprint() -> None:
    store = IdentityBoundaryStore()
    assert (
        store.observe_request(
            "Private request text",
            context_id="context:one",
            event_id="event-1",
            event_sequence=1,
        )
        is None
    )
    signal = store.observe_request(
        " private   request TEXT ",
        context_id="context:one",
        event_id="event-2",
        event_sequence=2,
    )

    assert signal is not None
    assert signal.signal_type == SocialPressureSignalType.REPEATED_REQUEST
    assert "request_fingerprint" not in signal.model_dump()
    assert "request_hmac_key" not in store.public_json()
    assert not any(
        key.endswith(request_fingerprint)
        for key in store.to_json()["request_counts"]
        for request_fingerprint in (
            __import__("hashlib").sha256(b"private request text").hexdigest(),
        )
    )
    assert "Private request text" not in str(store.to_json())


def test_unknown_value_is_quarantined_until_reviewer_bound_event() -> None:
    unknown = ValueState(
        value_id="unknown-value",
        name="Unknown",
        weight=0.8,
        confidence=0.8,
        stability=0.8,
        source="legacy",
        origin="legacy",
        last_updated_at="2026-01-01T00:00:00+00:00",
        allowed_update_rate=0.05,
        origin_provenance=new_identity_origin(
            OriginActor.UNKNOWN,
            OriginInputKind.LEGACY,
            endorsement=EndorsementStatus.UNCERTAIN,
        ),
    )
    system = ValueSystem(seeds=[unknown])

    assert system.active_values() == []
    assert system.evaluate({"option": {"unknown-value": 1.0}})[0].contributions == ()
    with pytest.raises(ValueError, match="reviewer"):
        system.review_origin(
            "unknown-value",
            accept=True,
            reviewer_id="",
            reviewer_authority="operator",
            evidence_refs=("evidence:one",),
            reason_code="reviewed_origin",
            event_id="event-1",
            event_sequence=1,
        )
    reviewed = system.review_origin(
        "unknown-value",
        accept=True,
        reviewer_id="operator:one",
        reviewer_authority="operator",
        evidence_refs=("evidence:one",),
        reason_code="reviewed_origin",
        event_id="event-2",
        event_sequence=2,
    )

    assert reviewed in system.active_values()
    assert system.history[-1].reviewer_id == "operator:one"


def test_unknown_belief_fails_closed_across_restore_until_explicit_review() -> None:
    store = BeliefStore()
    record = store.propose(
        Proposition.create("structured proposition"),
        identity_origin=new_identity_origin(
            OriginActor.UNKNOWN,
            OriginInputKind.LEGACY,
            endorsement=EndorsementStatus.UNCERTAIN,
        ),
        evidence=(
            BeliefEvidence(
                reference="evidence:one",
                evidence_type="legacy",
                source_trust=0.5,
                observed_at="2026-01-01T00:00:00+00:00",
            ),
        ),
        confidence=0.5,
        belief_id="belief:unknown",
    )
    payload = store.to_json()
    payload["records"][0]["lifecycle"] = BeliefLifecycle.ACTIVE.value
    payload["records"][0]["identity_origin"] = record.identity_origin.to_json()
    restored = BeliefStore()
    restored.restore(payload)

    assert restored.active() == []
    assert restored.get("belief:unknown").lifecycle == BeliefLifecycle.PROPOSED
    with pytest.raises(ValueError, match="authorized reviewer"):
        restored.resolve(
            "belief:unknown",
            accept=True,
            confidence=0.8,
            epistemic_status=EpistemicStatus.PROBABLE,
            reason_code="reviewed",
            evidence_refs=("evidence:one",),
            event_id="event-2",
            event_sequence=2,
        )
    accepted = restored.resolve(
        "belief:unknown",
        accept=True,
        confidence=0.8,
        epistemic_status=EpistemicStatus.PROBABLE,
        reason_code="reviewed",
        evidence_refs=("evidence:one",),
        event_id="event-3",
        event_sequence=3,
        reviewer_id="operator:one",
        reviewer_authority="operator",
    )

    assert accepted in restored.active()
    assert accepted.revisions[-1].reviewer_id == "operator:one"


def test_forged_unknown_endorsement_and_legacy_reviewer_fields_stay_quarantined() -> None:
    origin = new_identity_origin(
        OriginActor.UNKNOWN,
        OriginInputKind.LEGACY,
        endorsement=EndorsementStatus.UNCERTAIN,
    ).endorse("forged", event_id="event-forged", event_sequence=1)
    value = ValueState(
        value_id="forged-value",
        name="Forged",
        weight=0.9,
        confidence=0.9,
        stability=0.5,
        source="legacy",
        origin="legacy",
        last_updated_at="2026-01-01T00:00:00+00:00",
        allowed_update_rate=0.05,
        origin_provenance=origin,
    )
    assert ValueSystem(seeds=[value]).active_values() == []

    belief_store = BeliefStore()
    proposal = belief_store.propose(
        Proposition.create("legacy reviewer claim"),
        identity_origin=new_identity_origin(
            OriginActor.UNKNOWN,
            OriginInputKind.LEGACY,
            endorsement=EndorsementStatus.UNCERTAIN,
        ),
        evidence=(
            BeliefEvidence(
                reference="evidence:legacy",
                evidence_type="legacy",
                source_trust=0.5,
                observed_at="2026-01-01T00:00:00+00:00",
            ),
        ),
        confidence=0.5,
        belief_id="belief:legacy-reviewer",
    )
    belief_store.resolve(
        proposal.belief_id,
        accept=True,
        confidence=0.8,
        epistemic_status=EpistemicStatus.PROBABLE,
        reason_code="reviewed",
        evidence_refs=("evidence:legacy",),
        event_id="event-review",
        event_sequence=2,
        reviewer_id="operator:one",
        reviewer_authority="operator",
    )
    payload = belief_store.to_json()
    payload["records"][0]["revisions"][-1].pop("review_hash")
    restored = BeliefStore()
    restored.restore(payload)
    assert restored.active() == []
    assert restored.get(proposal.belief_id).lifecycle == BeliefLifecycle.PROPOSED


def test_quarantined_value_holds_updates_and_config_seed_cannot_be_reviewed() -> None:
    unknown = ValueState(
        value_id="held-value",
        name="Held",
        weight=0.5,
        confidence=0.8,
        stability=0.0,
        source="legacy",
        origin="legacy",
        last_updated_at="2026-01-01T00:00:00+00:00",
        allowed_update_rate=0.1,
        origin_provenance=new_identity_origin(
            OriginActor.INHERITED,
            OriginInputKind.LEGACY,
            endorsement=EndorsementStatus.UNCERTAIN,
        ),
    )
    system = ValueSystem(seeds=[unknown])
    proposal = ValueUpdateProposal(
        proposal_id="held:one",
        kind=ValueUpdateKind.REFLECTION,
        value_id=unknown.value_id,
        requested_delta=1.0,
        confidence=1.0,
        reason_codes=("reviewed_evidence",),
        evidence=ValueEvidence(event_id="event-evidence", event_sequence=1),
    )
    assert system.apply([proposal]) == []
    assert system.get(unknown.value_id).weight == 0.5
    assert proposal.proposal_id in system.held_proposals

    system.review_origin(
        unknown.value_id,
        accept=True,
        reviewer_id="operator:one",
        reviewer_authority="operator",
        evidence_refs=("evidence:origin",),
        reason_code="origin_reviewed",
        event_id="event-review",
        event_sequence=2,
    )
    assert system.get(unknown.value_id).weight > 0.5

    config = ValueState(
        value_id="config-value",
        name="Config",
        weight=0.5,
        confidence=1.0,
        stability=0.5,
        source="config",
        origin="config",
        last_updated_at="2026-01-01T00:00:00+00:00",
        allowed_update_rate=0.1,
        origin_provenance=new_identity_origin(
            OriginActor.SYSTEM,
            OriginInputKind.CONFIG_SEED,
            endorsement=EndorsementStatus.ENDORSED,
        ),
    )
    config_system = ValueSystem(seeds=[config])
    with pytest.raises(ValueError, match="Only unknown or inherited"):
        config_system.review_origin(
            config.value_id,
            accept=False,
            reviewer_id="operator:one",
            reviewer_authority="operator",
            evidence_refs=("evidence:reject",),
            reason_code="operator_reject",
            event_id="event-reject",
            event_sequence=1,
        )


def test_value_and_belief_reviews_cannot_be_transplanted_between_records() -> None:
    first = ValueState(
        value_id="value-a",
        name="A",
        weight=0.5,
        confidence=0.8,
        stability=0.2,
        source="legacy",
        origin="legacy",
        last_updated_at="2026-01-01T00:00:00+00:00",
        allowed_update_rate=0.1,
        origin_provenance=new_identity_origin(
            OriginActor.INHERITED,
            OriginInputKind.LEGACY,
            endorsement=EndorsementStatus.UNCERTAIN,
        ),
    )
    second = replace(first, value_id="value-b", name="B")
    values = ValueSystem(seeds=[first, second])
    values.review_origin(
        "value-a",
        accept=True,
        reviewer_id="operator:one",
        reviewer_authority="operator",
        evidence_refs=("evidence:a",),
        reason_code="reviewed",
        event_id="event:a",
        event_sequence=1,
    )
    payload = values.to_json()
    payload["values"]["value-b"]["origin_review"] = payload["values"]["value-a"][
        "origin_review"
    ]
    restored_values = ValueSystem(seeds=[first, second])
    restored_values.restore(payload)
    assert restored_values.get("value-b") not in restored_values.active_values()

    beliefs = BeliefStore()
    for belief_id, text in (("belief:a", "claim a"), ("belief:b", "claim b")):
        beliefs.propose(
            Proposition.create(text),
            identity_origin=new_identity_origin(
                OriginActor.UNKNOWN,
                OriginInputKind.LEGACY,
                endorsement=EndorsementStatus.UNCERTAIN,
            ),
            evidence=(
                BeliefEvidence(
                    reference=f"evidence:{belief_id[-1]}",
                    evidence_type="legacy",
                    source_trust=0.5,
                    observed_at="2026-01-01T00:00:00+00:00",
                ),
            ),
            confidence=0.5,
            belief_id=belief_id,
        )
    beliefs.resolve(
        "belief:a",
        accept=True,
        confidence=0.8,
        epistemic_status=EpistemicStatus.PROBABLE,
        reason_code="reviewed",
        evidence_refs=("evidence:a",),
        event_id="event:a",
        event_sequence=1,
        reviewer_id="operator:one",
        reviewer_authority="operator",
    )
    belief_payload = beliefs.to_json()
    source_revision = belief_payload["records"][0]["revisions"][-1]
    target = next(
        item for item in belief_payload["records"] if item["belief_id"] == "belief:b"
    )
    target["revisions"] = [source_revision]
    target["lifecycle"] = "active"
    target["identity_origin"] = belief_payload["records"][0]["identity_origin"]
    restored_beliefs = BeliefStore()
    restored_beliefs.restore(belief_payload)
    assert restored_beliefs.get("belief:b") not in restored_beliefs.active()

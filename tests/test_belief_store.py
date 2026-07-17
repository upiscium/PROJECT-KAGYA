from datetime import UTC, datetime, timedelta
import json

from kagya.belief import (
    BeliefEvidence,
    BeliefLifecycle,
    BeliefStore,
    EpistemicStatus,
    Proposition,
)
from kagya.identity import OriginActor, OriginInputKind, new_identity_origin


def test_external_claim_remains_proposal_until_reviewed() -> None:
    store = BeliefStore()
    proposal = _proposal(store, "sky-blue", "blue")

    assert proposal.lifecycle == BeliefLifecycle.PROPOSED
    assert store.active() == []
    accepted = store.resolve(
        proposal.belief_id,
        accept=True,
        confidence=0.9,
        epistemic_status=EpistemicStatus.ESTABLISHED,
        reason_code="evidence_reviewed",
        evidence_refs=("experience:one",),
        event_id="event-2",
        event_sequence=2,
    )

    assert accepted.lifecycle == BeliefLifecycle.ACTIVE
    assert accepted.identity_origin.actor == OriginActor.USER
    assert accepted.identity_origin.endorsement.value == "endorsed"
    assert store.active() == [accepted]


def test_contradiction_is_tracked_without_overwriting_existing_belief() -> None:
    store = BeliefStore()
    first = _proposal(store, "door-open", "open")
    store.resolve(
        first.belief_id,
        accept=True,
        confidence=0.8,
        epistemic_status=EpistemicStatus.ESTABLISHED,
        reason_code="observed",
        evidence_refs=("experience:one",),
        event_id=None,
        event_sequence=None,
    )

    conflicting = _proposal(store, "door-closed", "closed")

    assert conflicting.lifecycle == BeliefLifecycle.DISPUTED
    assert store.get(first.belief_id).lifecycle == BeliefLifecycle.DISPUTED
    assert conflicting.contradiction_ids == (first.belief_id,)
    assert store.active() == []

    reviewed_conflict = store.resolve(
        conflicting.belief_id,
        accept=True,
        confidence=0.9,
        epistemic_status=EpistemicStatus.ESTABLISHED,
        reason_code="newer_evidence",
        evidence_refs=("experience:two",),
        event_id=None,
        event_sequence=None,
    )
    assert reviewed_conflict.lifecycle == BeliefLifecycle.DISPUTED
    _, replacement = store.supersede(
        first.belief_id,
        conflicting.belief_id,
        reason_code="newer_observation",
        evidence_refs=("experience:two",),
        event_id=None,
        event_sequence=None,
    )
    assert replacement.lifecycle == BeliefLifecycle.ACTIVE
    assert store.active() == [replacement]


def test_rejecting_conflict_reactivates_reviewed_existing_belief() -> None:
    store = BeliefStore()
    existing = _accepted(store, "existing", "open")
    conflict = _proposal(store, "conflict", "closed")

    rejected = store.resolve(
        conflict.belief_id,
        accept=False,
        confidence=0.1,
        epistemic_status=EpistemicStatus.UNCERTAIN,
        reason_code="claim_disproved",
        evidence_refs=("experience:correction",),
        event_id=None,
        event_sequence=None,
    )

    assert rejected.lifecycle == BeliefLifecycle.REJECTED
    reactivated = store.get(existing.belief_id)
    assert reactivated.lifecycle == BeliefLifecycle.ACTIVE
    assert reactivated.contradiction_ids == ()
    assert reactivated.revisions[-1].operation == "contradiction_resolved"


def test_superseded_retracted_and_expired_beliefs_are_not_active() -> None:
    store = BeliefStore()
    old = _accepted(store, "old", "open")
    replacement = _accepted(store, "replacement", "closed", subject="window")
    store.supersede(
        old.belief_id,
        replacement.belief_id,
        reason_code="newer_observation",
        evidence_refs=("experience:new",),
        event_id=None,
        event_sequence=None,
    )
    retracted = _accepted(store, "retracted", "present", subject="item")
    store.retract(
        retracted.belief_id,
        reason_code="operator_correction",
        evidence_refs=("memory:correction",),
        event_id=None,
        event_sequence=None,
    )
    expired = _accepted(
        store,
        "temporary",
        "available",
        subject="service",
        valid_until=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    )
    store.expire()

    active_ids = {record.belief_id for record in store.active()}
    assert old.belief_id not in active_ids
    assert retracted.belief_id not in active_ids
    assert expired.belief_id not in active_ids
    assert replacement.belief_id in active_ids


def test_context_scope_and_revision_history_round_trip() -> None:
    store = BeliefStore()
    accepted = _accepted(
        store,
        "scoped",
        "active",
        subject="task",
        context_scope=("ctx-one",),
    )
    payload = json.loads(json.dumps(store.to_json()))
    restored = BeliefStore()
    restored.restore(payload)

    assert restored.active(context_id="ctx-two") == []
    assert restored.active(context_id="ctx-one")[0].belief_id == accepted.belief_id
    assert restored.get(accepted.belief_id).revisions[-1].operation == "accept"
    assert restored.to_json() == payload


def test_external_claim_and_agent_inference_keep_distinct_origins() -> None:
    store = BeliefStore()
    external = _proposal(store, "external", "open")
    inference = store.propose(
        Proposition.create("A pattern may hold"),
        identity_origin=new_identity_origin(
            OriginActor.MODEL_INFERENCE,
            OriginInputKind.EVIDENCE,
            source_ref="decision:one",
        ),
        evidence=(
            BeliefEvidence(
                reference="decision:one",
                evidence_type="agent_inference",
                source_trust=0.5,
                observed_at=datetime.now(UTC).isoformat(),
            ),
        ),
        confidence=0.4,
        belief_id="inference",
    )

    assert external.identity_origin.actor == OriginActor.USER
    assert external.evidence[0].evidence_type == "external_claim"
    assert inference.identity_origin.actor == OriginActor.MODEL_INFERENCE
    assert inference.evidence[0].evidence_type == "agent_inference"


def _proposal(
    store: BeliefStore,
    belief_id: str,
    object_value: str,
    *,
    subject: str = "door",
    context_scope: tuple[str, ...] = (),
    valid_until: str | None = None,
):
    return store.propose(
        Proposition.create(
            f"{subject} is {object_value}",
            subject=subject,
            predicate="state",
            object=object_value,
        ),
        identity_origin=new_identity_origin(
            OriginActor.USER,
            OriginInputKind.OBSERVATION,
            source_ref="experience:one",
        ),
        evidence=(
            BeliefEvidence(
                reference="experience:one",
                evidence_type="external_claim",
                source_trust=0.7,
                observed_at=datetime.now(UTC).isoformat(),
            ),
        ),
        confidence=0.6,
        context_scope=context_scope,
        valid_until=valid_until,
        belief_id=belief_id,
    )


def _accepted(
    store: BeliefStore,
    belief_id: str,
    object_value: str,
    *,
    subject: str = "door",
    context_scope: tuple[str, ...] = (),
    valid_until: str | None = None,
):
    proposal = _proposal(
        store,
        belief_id,
        object_value,
        subject=subject,
        context_scope=context_scope,
        valid_until=valid_until,
    )
    return store.resolve(
        proposal.belief_id,
        accept=True,
        confidence=0.85,
        epistemic_status=EpistemicStatus.ESTABLISHED,
        reason_code="reviewed",
        evidence_refs=("experience:one",),
        event_id=None,
        event_sequence=None,
    )

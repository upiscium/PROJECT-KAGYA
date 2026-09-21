import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from kagya.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
)
from kagya.identity.value_system import (
    ValueConflictDefinition,
    ValueDomainError,
    ValueEvidence,
    ValueMutationEvidence,
    ValueMutationReason,
    ValueMutationStatus,
    ValueOriginReviewDecision,
    ValueProposal,
    ValueReason,
    ValueRevisionHistory,
    ValueRevisionOperation,
    ValueRevisionRecord,
    ValueSeedDeclaration,
    ValueSelfAdmission,
    ValueScope,
    ValueState,
    ValueSystem,
    canonical_seed_payload,
    canonical_value_state_payload,
    evidence_ledger_digest,
    recompute_revision_record_digest,
    recompute_seed_contract_digest,
    validate_revision_record_digest,
    validate_seed_contract_digest,
    value_state_digest,
)


def _origin(
    *, system: bool = False, event_id: str = "event-1", event_sequence: int = 0
) -> IdentityOrigin:
    if system:
        return IdentityOrigin(
            OriginActor.SYSTEM,
            OriginInputKind.CONFIG_SEED,
            ValueAdmissionStatus.SYSTEM_AUTHORIZED,
        )
    return IdentityOrigin(
        OriginActor.SELF,
        OriginInputKind.INTERNAL_STATE,
        ValueAdmissionStatus.SELF_ENDORSED,
        event_id=event_id,
        event_sequence=event_sequence,
    )


def _value(**changes: object) -> ValueState:
    fields: dict[str, object] = {
        "value_id": "value-1",
        "revision": 0,
        "name": "care",
        "concept": "Protect the wellbeing of the subject.",
        "scope": ValueScope.SUBJECT,
        "context_ids": (),
        "polarity": 1,
        "strength": 0.8,
        "confidence": 0.9,
        "stability": 0.7,
        "protectedness": 0.6,
        "negotiability": 0.2,
        "allowed_update_rate": 0.1,
        "frozen": False,
        "origin": _origin(),
        "evidence_refs": ("evidence-1",),
    }
    fields.update(changes)
    return ValueState(**fields)  # type: ignore[arg-type]


def _seed(**changes: object) -> ValueSeedDeclaration:
    fields: dict[str, object] = {
        "value_id": "value-1",
        "name": "care",
        "concept": "Protect the wellbeing of the subject.",
        "scope": ValueScope.SUBJECT,
        "context_ids": (),
        "polarity": 1,
        "initial_strength": 0.8,
        "confidence": 0.9,
        "stability": 0.7,
        "protectedness": 0.6,
        "negotiability": 0.2,
        "allowed_update_rate": 0.1,
    }
    fields.update(changes)
    return ValueSeedDeclaration(**fields)  # type: ignore[arg-type]


def _event(event_id: str = "event-1", event_sequence: int = 0) -> ValueMutationEvidence:
    return ValueMutationEvidence(
        event_id=event_id,
        event_sequence=event_sequence,
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        + timedelta(seconds=event_sequence),
    )


def _governance_origin(
    evidence: ValueMutationEvidence, source: str
) -> IdentityOrigin:
    return IdentityOrigin(
        OriginActor.OPERATOR,
        OriginInputKind.CONSTRAINT,
        ValueAdmissionStatus.PENDING,
        source_ref=source,
        event_id=evidence.event_id,
        event_sequence=evidence.event_sequence,
    )


def _admission(
    *,
    value_id: str = "value-1",
    event_id: str = "event-1",
    event_sequence: int = 0,
    evidence_ref: str = "update-1",
    requested_delta: float = 1.0,
    confidence: float = 1.0,
    reason: ValueMutationReason = ValueMutationReason.ADMITTED_UPDATE,
) -> ValueSelfAdmission:
    return ValueSelfAdmission(
        target_value_id=value_id,
        subject_origin=_origin(
            event_id=event_id, event_sequence=event_sequence
        ),
        evidence_refs=(evidence_ref,),
        requested_delta=requested_delta,
        confidence=confidence,
        reason=reason,
    )


def _mutable_value(value_id: str = "value-1", **changes: object) -> ValueState:
    fields: dict[str, object] = {
        "value_id": value_id,
        "stability": 0.0,
        "protectedness": 0.0,
        "negotiability": 1.0,
        "allowed_update_rate": 0.1,
        "confidence": 1.0,
        "evidence_refs": (f"seed-{value_id}",),
    }
    fields.update(changes)
    return _value(**fields)


def test_scope_and_active_read_semantics() -> None:
    value = _value()
    assert value.is_active()
    assert value.applies_to(None)
    assert value.applies_to("context-1")
    assert value.applies_to("context-2")
    contextual = _value(scope=ValueScope.CONTEXT, context_ids=("context-1",))
    assert contextual.applies_to("context-1")
    assert not contextual.applies_to(None)
    assert not contextual.applies_to("context-2")
    with pytest.raises(ValueError):
        contextual.applies_to("bad..id")


@pytest.mark.parametrize(
    "context_ids",
    [
        (),
        ("context-2", "context-1"),
        ("context-1", "context-1"),
        tuple(f"context-{index}" for index in range(17)),
    ],
)
def test_context_scope_requires_bounded_sorted_unique_contexts(
    context_ids: tuple[str, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _value(scope=ValueScope.CONTEXT, context_ids=context_ids)


def test_subject_scope_rejects_stored_contexts() -> None:
    with pytest.raises(ValueError):
        _value(scope=ValueScope.SUBJECT, context_ids=("context-1",))


@pytest.mark.parametrize(
    "field", ["strength", "confidence", "stability", "protectedness", "negotiability"]
)
def test_scalar_bounds_are_strict(field: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _value(**{field: 2.0})
    with pytest.raises(TypeError):
        _value(**{field: 1})


@pytest.mark.parametrize("revision", [True, -1, 1.0, "1"])
def test_revision_requires_a_nonnegative_exact_integer(revision: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _value(revision=revision)
    assert _value(revision=0).revision == 0


@pytest.mark.parametrize("polarity", [True, 0, 1.0, "1"])
def test_polarity_requires_exactly_negative_or_positive_one(polarity: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _value(polarity=polarity)
    assert _value(polarity=-1).polarity == -1
    assert _value(polarity=1).polarity == 1


def test_refs_conflicts_and_non_authoritative_types() -> None:
    with pytest.raises(ValueError):
        _value(evidence_refs=("evidence-2", "evidence-1"))
    with pytest.raises(ValueError):
        _value(evidence_refs=tuple(f"evidence-{index}" for index in range(17)))
    with pytest.raises(ValueError):
        ValueConflictDefinition("value-2", "value-1")
    with pytest.raises(ValueError):
        ValueConflictDefinition("value-1", "value-1")
    evidence = ValueEvidence("evidence-2", _origin(), ValueReason.OBSERVATION)
    proposal = ValueProposal(
        "proposal-1", _origin(), ValueReason.PROPOSAL, target_value_id="value-1"
    )
    assert evidence.evidence_ref == "evidence-2"
    assert proposal.target_value_id == "value-1"


def test_seed_digest_golden_and_changed_field() -> None:
    seed = _seed()
    payload = b'kagya.identity.value-seed/v1\x00{"allowed_update_rate":0.1,"concept":"Protect the wellbeing of the subject.","confidence":0.9,"context_ids":[],"initial_strength":0.8,"name":"care","negotiability":0.2,"polarity":1,"protectedness":0.6,"scope":"subject","stability":0.7,"value_id":"value-1"}'
    expected = "081ba5fbfc1075cfa2246790e137080f9eb500e4ecf1e3f42b6c322ef641e146"
    assert canonical_seed_payload(seed) == payload
    assert expected == hashlib.sha256(payload).hexdigest()
    assert validate_seed_contract_digest(seed, expected) == expected
    assert recompute_seed_contract_digest(seed) == expected
    with pytest.raises(TypeError):
        canonical_seed_payload(_value())  # type: ignore[arg-type]


def test_optional_concept_is_canonicalized_as_json_null() -> None:
    seed = _seed(concept=None)
    assert b'"concept":null' in canonical_seed_payload(seed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value_id", "value-2"),
        ("name", "different"),
        ("concept", "A different concept."),
        ("polarity", -1),
        ("initial_strength", 0.7),
        ("confidence", 0.8),
        ("stability", 0.6),
        ("protectedness", 0.5),
        ("negotiability", 0.3),
        ("allowed_update_rate", 0.2),
    ],
)
def test_each_seed_scalar_changes_contract_digest(field: str, value: object) -> None:
    assert recompute_seed_contract_digest(_seed(**{field: value})) != recompute_seed_contract_digest(
        _seed()
    )


def test_seed_scope_and_context_ids_change_contract_digest() -> None:
    subject = _seed()
    contextual = _seed(scope=ValueScope.CONTEXT, context_ids=("context-1",))
    another_context = _seed(scope=ValueScope.CONTEXT, context_ids=("context-2",))
    assert recompute_seed_contract_digest(contextual) != recompute_seed_contract_digest(subject)
    assert recompute_seed_contract_digest(another_context) != recompute_seed_contract_digest(
        contextual
    )


def test_system_seed_digest_requires_a_valid_shape() -> None:
    seed = _seed()
    digest = recompute_seed_contract_digest(seed)
    system_value = _value(
        origin=_origin(system=True),
        seed_contract_digest=digest,
    )
    assert system_value.is_active()
    nullable_concept_seed = _seed(concept=None)
    assert _value(
        concept=None,
        origin=_origin(system=True),
        seed_contract_digest=recompute_seed_contract_digest(nullable_concept_seed),
    ).is_active()
    assert _value(
        origin=_origin(system=True),
        seed_contract_digest="0" * 64,
    ).seed_contract_digest == "0" * 64
    for malformed in ("", "0" * 63, "0" * 63 + "G"):
        with pytest.raises(ValueError):
            _value(origin=_origin(system=True), seed_contract_digest=malformed)


def test_system_value_can_evolve_without_rewriting_seed_digest() -> None:
    seed_digest = recompute_seed_contract_digest(_seed())
    bootstrap = _value(
        revision=0,
        strength=0.8,
        confidence=0.9,
        origin=_origin(system=True),
        seed_contract_digest=seed_digest,
    )
    learned = _value(
        revision=1,
        strength=0.79,
        confidence=0.88,
        origin=_origin(system=True),
        seed_contract_digest=seed_digest,
    )
    assert bootstrap.seed_contract_digest == learned.seed_contract_digest == seed_digest
    assert bootstrap.strength != learned.strength
    assert bootstrap.confidence != learned.confidence


def test_from_seed_declarations_is_deterministic_and_preserves_seed_lineage() -> None:
    care = _seed(value_id="care", name="care")
    honesty = _seed(
        value_id="honesty", name="honesty", concept="Represent what is true."
    )
    conflicts = (ValueConflictDefinition("care", "honesty"),)

    first = ValueSystem.from_seed_declarations((honesty, care), conflicts)
    second = ValueSystem.from_seed_declarations((care, honesty), conflicts)

    assert first.values == second.values
    assert first.conflicts == conflicts
    for seed in (care, honesty):
        value = first.get(seed.value_id)
        assert value.seed_contract_digest == recompute_seed_contract_digest(seed)
        assert value.origin.actor is OriginActor.SYSTEM
        assert value.origin.input_kind is OriginInputKind.CONFIG_SEED
        assert value.origin.admission is ValueAdmissionStatus.SYSTEM_AUTHORIZED
        assert value.origin.source_ref == f"config-seed:{seed.value_id}"
        assert value.origin.event_id is None
        assert value.origin.event_sequence is None
        assert value.revision == 0
        assert not value.frozen
        assert value.opposition_count == 0
        assert value.evidence_refs == ()
        assert first.applied_evidence(seed.value_id) == ()


def test_from_seed_declarations_uses_domain_authority_for_conflict_ids() -> None:
    with pytest.raises(ValueDomainError):
        ValueSystem.from_seed_declarations(
            (_seed(value_id="care"),),
            (ValueConflictDefinition("care", "honesty"),),
        )


@pytest.mark.parametrize(
    "admission",
    [
        ValueAdmissionStatus.PENDING,
        ValueAdmissionStatus.REJECTED,
        ValueAdmissionStatus.UNCERTAIN,
    ],
)
def test_inactive_admissions_are_not_active(admission: ValueAdmissionStatus) -> None:
    origin = IdentityOrigin(
        OriginActor.USER,
        OriginInputKind.EVIDENCE,
        admission,
    )
    assert not _value(origin=origin).is_active()


def test_system_and_self_admissions_are_active() -> None:
    self_value = _value()
    system_value = _value(
        origin=_origin(system=True),
        seed_contract_digest=recompute_seed_contract_digest(_seed()),
    )
    assert self_value.is_active()
    assert system_value.is_active()


def test_private_raw_text_cannot_enter_opaque_evidence_references() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValueEvidence("PRIVATE SENTINEL\n", _origin(), ValueReason.OBSERVATION)
    with pytest.raises((TypeError, ValueError)):
        ValueEvidence(
            "evidence-1",
            _origin(),
            ValueReason.OBSERVATION,
            target_value_id="PRIVATE SENTINEL",
        )


def test_evidence_and_proposal_do_not_mutate_value_state() -> None:
    value = _value()
    before = replace(value)
    evidence = ValueEvidence("evidence-2", _origin(), ValueReason.OBSERVATION)
    proposal = ValueProposal("proposal-1", _origin(), ValueReason.PROPOSAL)
    assert evidence and proposal
    assert value == before
    assert not hasattr(proposal, "apply")


def test_mutation_evidence_requires_canonical_utc() -> None:
    assert _event().recorded_at.tzinfo is timezone.utc
    with pytest.raises(ValueError):
        ValueMutationEvidence("event-1", 0, datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        ValueMutationEvidence(
            "event-1", 0, datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1)))
        )
    with pytest.raises(TypeError):
        ValueMutationEvidence("event-1", True, _event().recorded_at)  # type: ignore[arg-type]


def test_self_admission_requires_exact_event_binding_and_candidate_shape() -> None:
    system = ValueSystem()
    candidate = _value(evidence_refs=("admission-1",))
    admission = _admission(
        reason=ValueMutationReason.SELF_ADMISSION,
        evidence_ref="admission-1",
    )
    result = system.admit_self_value(candidate, admission, _event())
    assert result.status is ValueMutationStatus.APPLIED
    assert system.get("value-1") == candidate
    assert system.history("value-1").records[0].operation.value == "admission"
    assert system.history("value-1").records[0].from_revision == -1
    retry = system.admit_self_value(candidate, admission, _event())
    assert retry.status is ValueMutationStatus.IDEMPOTENT

    with pytest.raises(ValueDomainError):
        system.admit_self_value(candidate, admission, _event("event-2", 1))
    with pytest.raises(ValueDomainError):
        system.admit_self_value(replace(candidate, strength=0.7), admission, _event())

    with pytest.raises(ValueDomainError):
        ValueSystem((_value(revision=1),))

    external = IdentityOrigin(
        OriginActor.USER,
        OriginInputKind.EVIDENCE,
        ValueAdmissionStatus.PENDING,
    )
    with pytest.raises(ValueError):
        ValueSelfAdmission(
            "value-2",
            external,
            ("admission-2",),
            0.0,
            1.0,
            ValueMutationReason.SELF_ADMISSION,
        )


def test_system_update_uses_per_value_cap_and_canonical_global_budget() -> None:
    first = _mutable_value("value-1")
    second = _mutable_value("value-2")
    system = ValueSystem((second, first))
    result = system.apply_updates(
        _event(),
        (
            _admission(value_id="value-2", evidence_ref="update-2"),
            _admission(value_id="value-1", evidence_ref="update-1"),
        ),
    )
    assert tuple(item.value_id for item in result.results) == ("value-1", "value-2")
    assert result[0].applied_delta == pytest.approx(0.1)
    assert result[1].applied_delta == pytest.approx(0.0)
    assert abs(result.total_applied_delta) <= 0.1
    assert system.get("value-1").revision == 1
    assert system.get("value-2").revision == 0


def test_zero_cap_records_evidence_without_creating_a_revision() -> None:
    value = _mutable_value(negotiability=0.0)
    system = ValueSystem((value,))
    result = system.apply_update(_admission(), _event())
    assert result.status is ValueMutationStatus.NO_CHANGE
    assert result.value == value
    assert system.get("value-1").revision == 0
    assert system.history("value-1").records == ()
    assert system.applied_evidence("value-1") == ("seed-value-1", "update-1")


def test_evidence_ledger_overflow_fails_without_mutation() -> None:
    system = ValueSystem((_mutable_value(),))
    for index in range(31):
        event_id = f"ledger-event-{index}"
        refs = tuple(f"ledger-{index:02d}-{slot:02d}" for slot in range(16))
        system.apply_update(
            ValueSelfAdmission(
                "value-1",
                _origin(event_id=event_id, event_sequence=index),
                refs,
                0.0,
                1.0,
                ValueMutationReason.ADMITTED_UPDATE,
            ),
            _event(event_id, index),
        )
    before_value = system.get("value-1")
    before_ledger = system.applied_evidence("value-1")
    refs = tuple(f"overflow-{slot:02d}" for slot in range(16))
    with pytest.raises(ValueDomainError):
        system.apply_update(
            ValueSelfAdmission(
                "value-1",
                _origin(event_id="ledger-overflow", event_sequence=31),
                refs,
                0.0,
                1.0,
                ValueMutationReason.ADMITTED_UPDATE,
            ),
            _event("ledger-overflow", 31),
        )
    assert system.get("value-1") == before_value
    assert system.applied_evidence("value-1") == before_ledger


def test_support_and_opposition_policy_is_bounded_and_deterministic() -> None:
    value = _mutable_value(strength=0.05)
    system = ValueSystem((value,))
    first_opposition = system.apply_update(
        _admission(requested_delta=-1.0, evidence_ref="opp-1"), _event()
    )
    assert first_opposition.status is ValueMutationStatus.APPLIED
    assert first_opposition.value.strength == pytest.approx(0.0)
    assert first_opposition.value.polarity == 1
    assert first_opposition.value.opposition_count == 1

    for index in range(2, 4):
        result = system.apply_update(
            _admission(
                event_id=f"event-{index}",
                event_sequence=index - 1,
                evidence_ref=f"opp-{index}",
                requested_delta=-1.0,
            ),
            _event(f"event-{index}", index - 1),
        )
        assert result.value.polarity == 1
    reversal = system.apply_update(
        _admission(
            event_id="event-4",
            event_sequence=3,
            evidence_ref="opp-4",
            requested_delta=-1.0,
        ),
        _event("event-4", 3),
    )
    assert reversal.value.polarity == -1
    assert reversal.value.opposition_count == 0

    support = system.apply_update(
        _admission(
            event_id="event-5",
            event_sequence=4,
            evidence_ref="support-1",
            requested_delta=1.0,
        ),
        _event("event-5", 4),
    )
    assert support.value.opposition_count == 0
    assert support.value.confidence > reversal.value.confidence


def test_protectedness_raises_reversal_threshold() -> None:
    value = _mutable_value(strength=0.0, protectedness=0.9)
    system = ValueSystem((value,))
    for index in range(1, 7):
        event_id = f"protected-event-{index}"
        result = system.apply_update(
            _admission(
                event_id=event_id,
                event_sequence=index,
                evidence_ref=f"protected-{index}",
                requested_delta=-1.0,
            ),
            _event(event_id, index),
        )
        assert result.value.polarity == 1
    result = system.apply_update(
        _admission(
            event_id="protected-event-7",
            event_sequence=7,
            evidence_ref="protected-7",
            requested_delta=-1.0,
        ),
        _event("protected-event-7", 7),
    )
    assert result.value.polarity == -1


def test_duplicate_targets_and_duplicate_evidence_are_fail_closed() -> None:
    value = _mutable_value()
    system = ValueSystem((value,))
    duplicate = _admission(evidence_ref="duplicate-1")
    with pytest.raises(ValueDomainError):
        system.apply_updates(_event(), (duplicate, duplicate))
    first = system.apply_update(duplicate, _event())
    before = system.get("value-1")
    history = system.history("value-1")
    duplicate_result = system.apply_update(duplicate, _event())
    assert first.status is ValueMutationStatus.APPLIED
    assert duplicate_result.status is ValueMutationStatus.IDEMPOTENT
    assert system.get("value-1") == before
    assert system.history("value-1") == history


def test_replayed_evidence_makes_the_whole_event_idempotent() -> None:
    first = _mutable_value("value-1")
    second = _mutable_value("value-2")
    system = ValueSystem((first, second))
    system.apply_update(_admission(evidence_ref="replayed-1"), _event())
    before_second = system.get("value-2")
    result = system.apply_updates(
        _event("event-2", 1),
        (
            _admission(
                event_id="event-2",
                event_sequence=1,
                evidence_ref="replayed-1",
            ),
            _admission(
                value_id="value-2",
                event_id="event-2",
                event_sequence=1,
                evidence_ref="new-2",
            ),
        ),
    )
    assert all(item.status is ValueMutationStatus.IDEMPOTENT for item in result.results)
    assert system.get("value-2") == before_second


def test_freeze_unfreeze_and_frozen_updates_have_no_hidden_mutation() -> None:
    value = _mutable_value()
    system = ValueSystem((value,))
    frozen_event = _event()
    frozen = system.freeze(
        "value-1",
        frozen_event,
        governance_origin=_governance_origin(frozen_event, "api.values.freeze"),
    )
    assert frozen.status is ValueMutationStatus.APPLIED
    before_frozen = system.get("value-1")
    before_history = system.history("value-1")
    before_ledger = system.applied_evidence("value-1")
    blocked = system.apply_update(
        _admission(event_id="event-2", event_sequence=1, evidence_ref="blocked-1"),
        _event("event-2", 1),
    )
    assert blocked.status is ValueMutationStatus.FROZEN
    assert system.get("value-1") == before_frozen
    assert system.history("value-1") == before_history
    assert system.applied_evidence("value-1") == before_ledger
    idempotent_event = _event("event-3", 2)
    assert (
        system.freeze(
            "value-1",
            idempotent_event,
            governance_origin=_governance_origin(
                idempotent_event, "api.values.freeze"
            ),
        ).status
        is ValueMutationStatus.IDEMPOTENT
    )
    unfrozen_event = _event("event-4", 3)
    unfrozen = system.unfreeze(
        "value-1",
        unfrozen_event,
        governance_origin=_governance_origin(unfrozen_event, "api.values.unfreeze"),
    )
    assert unfrozen.status is ValueMutationStatus.APPLIED
    assert unfrozen.value.revision == before_frozen.revision + 1
    retry_event = _event("event-5", 4)
    assert (
        system.unfreeze(
            "value-1",
            retry_event,
            governance_origin=_governance_origin(retry_event, "api.values.unfreeze"),
        ).status
        is ValueMutationStatus.IDEMPOTENT
    )


def test_governance_freeze_accepts_quarantined_values_and_binds_origin() -> None:
    quarantine = _value(
        origin=IdentityOrigin(
            OriginActor.INHERITED,
            OriginInputKind.LEGACY,
            ValueAdmissionStatus.UNCERTAIN,
        ),
        evidence_refs=(),
    )
    system = ValueSystem((quarantine,))
    evidence = _event("freeze-quarantine", 7)
    result = system.freeze(
        "value-1",
        evidence,
        governance_origin=_governance_origin(evidence, "api.values.freeze"),
    )

    assert result.status is ValueMutationStatus.APPLIED
    assert not result.value.is_active()
    assert result.revision_record is not None
    assert result.revision_record.origin_id == _governance_origin(
        evidence, "api.values.freeze"
    ).origin_id
    with pytest.raises(ValueDomainError):
        system.freeze(
            "value-1",
            evidence,
            governance_origin=_governance_origin(evidence, "api.values.rollback"),
        )


def test_config_seed_adoption_is_exact_idempotent_and_collision_safe() -> None:
    seed = _seed(value_id="new-seed", name="new-seed")
    system = ValueSystem()
    evidence = _event("seed-adopt", 3)
    origin = _governance_origin(evidence, "api.values.seed_adopt")
    result = system.adopt_seed(seed, evidence, governance_origin=origin)

    adopted = system.get("new-seed")
    assert result.status is ValueMutationStatus.APPLIED
    assert adopted.origin.actor is OriginActor.SYSTEM
    assert adopted.origin.input_kind is OriginInputKind.CONFIG_SEED
    assert adopted.origin.admission is ValueAdmissionStatus.SYSTEM_AUTHORIZED
    assert adopted.seed_contract_digest == recompute_seed_contract_digest(seed)
    assert result.revision_record is not None
    assert result.revision_record.operation is ValueRevisionOperation.ADMISSION
    assert result.revision_record.origin_id == origin.origin_id
    assert system.conflicts == ()

    retry = system.adopt_seed(
        seed,
        _event("seed-adopt-retry", 4),
        governance_origin=_governance_origin(
            _event("seed-adopt-retry", 4), "api.values.seed_adopt"
        ),
    )
    assert retry.status is ValueMutationStatus.IDEMPOTENT

    with pytest.raises(ValueDomainError):
        system.adopt_seed(
            _seed(value_id="new-seed", name="changed"),
            _event("seed-collision", 5),
            governance_origin=_governance_origin(
                _event("seed-collision", 5), "api.values.seed_adopt"
            ),
        )

    non_system = ValueSystem((_value(value_id="new-seed", evidence_refs=()),))
    with pytest.raises(ValueDomainError):
        non_system.adopt_seed(
            seed,
            _event("seed-non-system", 6),
            governance_origin=_governance_origin(
                _event("seed-non-system", 6), "api.values.seed_adopt"
            ),
        )


@pytest.mark.parametrize(
    ("actor", "initial_admission"),
    [
        (OriginActor.INHERITED, ValueAdmissionStatus.UNCERTAIN),
        (OriginActor.UNKNOWN, ValueAdmissionStatus.UNCERTAIN),
    ],
)
def test_origin_review_preserves_lineage_and_never_endorses(
    actor: OriginActor, initial_admission: ValueAdmissionStatus
) -> None:
    original_origin = IdentityOrigin(
        actor, OriginInputKind.LEGACY, initial_admission
    )
    system = ValueSystem(
        (_value(origin=original_origin, evidence_refs=()),)
    )
    accept_event = _event(f"review-accept-{actor.value}", 8)
    accepted = system.review_origin(
        "value-1",
        ValueOriginReviewDecision.ACCEPT_PROVENANCE,
        accept_event,
        governance_origin=_governance_origin(
            accept_event, "api.values.origin_review"
        ),
    )
    assert accepted.value.origin.origin_id == original_origin.origin_id
    assert accepted.value.origin.admission is ValueAdmissionStatus.PENDING
    assert accepted.value.origin.actor is actor
    assert not accepted.value.is_active()
    assert accepted.revision_record is not None
    assert accepted.revision_record.operation is ValueRevisionOperation.ORIGIN_REVIEW
    assert accepted.revision_record.origin_id != original_origin.origin_id

    reject_event = _event(f"review-reject-{actor.value}", 9)
    rejected = system.review_origin(
        "value-1",
        ValueOriginReviewDecision.REJECT,
        reject_event,
        governance_origin=_governance_origin(
            reject_event, "api.values.origin_review"
        ),
    )
    assert rejected.value.origin.origin_id == original_origin.origin_id
    assert rejected.value.origin.admission is ValueAdmissionStatus.REJECTED
    assert not rejected.value.is_active()

    retry = system.review_origin(
        "value-1",
        ValueOriginReviewDecision.REJECT,
        _event("review-reject-retry", 10),
        governance_origin=_governance_origin(
            _event("review-reject-retry", 10), "api.values.origin_review"
        ),
    )
    assert retry.status is ValueMutationStatus.IDEMPOTENT

    with pytest.raises(ValueDomainError):
        system.review_origin(
            "value-1",
            ValueOriginReviewDecision.ACCEPT_PROVENANCE,
            _event("review-reopen", 11),
            governance_origin=_governance_origin(
                _event("review-reopen", 11), "api.values.origin_review"
            ),
        )


def test_origin_review_rejects_self_and_system_lineage() -> None:
    system_value = _value(
        value_id="system-value",
        origin=_origin(system=True),
        seed_contract_digest=recompute_seed_contract_digest(_seed()),
        evidence_refs=(),
    )
    system = ValueSystem((_value(), system_value))
    for value_id, event_id, sequence in (
        ("value-1", "review-self", 12),
        ("system-value", "review-system", 13),
    ):
        event = _event(event_id, sequence)
        with pytest.raises(ValueDomainError):
            system.review_origin(
                value_id,
                ValueOriginReviewDecision.REJECT,
                event,
                governance_origin=_governance_origin(
                    event, "api.values.origin_review"
                ),
            )


def test_revision_state_and_record_digests_are_canonical() -> None:
    value = _mutable_value()
    payload = canonical_value_state_payload(value)
    expected_payload = (
        b'kagya.identity.value-state/v1\x00{"allowed_update_rate":0.1,"concept":"Protect the wellbeing of the subject.",'
        b'"confidence":1.0,"context_ids":[],'
        b'"evidence_refs":["seed-value-1"],"frozen":false,"name":"care",'
        b'"negotiability":1.0,"opposition_count":0,"origin":{"actor":"self","admission":"self_endorsed",'
        b'"confidence":1.0,"context_id":null,"event_id":"event-1",'
        b'"event_sequence":0,"input_kind":"internal_state","origin_id":"'
        + value.origin.origin_id.encode("ascii")
        + b'","source_ref":null},"polarity":1,"protectedness":0.0,'
        b'"revision":0,"scope":"subject",'
        b'"seed_contract_digest":null,"stability":0.0,"strength":0.8,'
        b'"value_id":"value-1"}'
    )
    assert payload == expected_payload
    assert value_state_digest(value) == hashlib.sha256(payload).hexdigest()

    system = ValueSystem((value,))
    result = system.apply_update(_admission(), _event())
    assert result.revision_record is not None
    record = result.revision_record
    assert validate_revision_record_digest(record) == record.record_digest
    assert recompute_revision_record_digest(record) == record.record_digest
    tampered = record
    object.__setattr__(tampered, "after_digest", "0" * 64)
    with pytest.raises(ValueError):
        validate_revision_record_digest(tampered)


def test_revision_history_compacts_without_rewriting_chain_or_ledger() -> None:
    value = _mutable_value()
    system = ValueSystem((value,))
    for index in range(33):
        event_id = f"chain-event-{index}"
        system.apply_update(
            _admission(
                event_id=event_id,
                event_sequence=index,
                evidence_ref=f"chain-{index}",
            ),
            _event(event_id, index),
        )
    history = system.history("value-1")
    assert len(history.records) == 32
    assert history.history_anchor_revision == 1
    assert history.history_anchor_state_digest == history.records[0].before_digest
    assert history.records[0].from_revision == 1
    assert history.records[0].previous_record_digest == history.history_anchor_digest
    assert len(system.applied_evidence("value-1")) == 34
    history.validate()

    current = system.get("value-1")
    rollback_event = _event("rollback-event", 100)
    rolled_back = system.rollback(
        "value-1",
        2,
        rollback_event,
        governance_origin=_governance_origin(rollback_event, "api.values.rollback"),
    )
    assert rolled_back.status is ValueMutationStatus.APPLIED
    assert rolled_back.value.revision == current.revision + 1
    assert rolled_back.value.origin == current.origin
    assert rolled_back.value.seed_contract_digest == current.seed_contract_digest
    assert rolled_back.revision_record is not None
    assert rolled_back.revision_record.target_revision == 2
    assert len(system.applied_evidence("value-1")) == 34
    with pytest.raises(ValueDomainError):
        old_rollback_event = _event("old-rollback", 101)
        system.rollback(
            "value-1",
            1,
            old_rollback_event,
            governance_origin=_governance_origin(
                old_rollback_event, "api.values.rollback"
            ),
        )


def test_system_authorized_value_retains_seed_lineage_during_updates() -> None:
    digest = recompute_seed_contract_digest(_seed())
    system_value = _mutable_value(
        origin=_origin(system=True),
        seed_contract_digest=digest,
    )
    system = ValueSystem((system_value,))
    result = system.apply_update(_admission(), _event())
    assert result.value.seed_contract_digest == digest
    assert result.value.strength != system_value.strength
    assert result.value.origin == system_value.origin


@pytest.mark.parametrize("actor", [OriginActor.INHERITED, OriginActor.UNKNOWN])
def test_value_system_retains_inactive_quarantine_values(actor: OriginActor) -> None:
    value_id = f"quarantine-{actor.value}"
    quarantine = _value(
        value_id=value_id,
        origin=IdentityOrigin(actor, OriginInputKind.LEGACY, ValueAdmissionStatus.UNCERTAIN),
        evidence_refs=(),
    )
    active = _mutable_value()
    system = ValueSystem(
        (active, quarantine),
        conflicts=(ValueConflictDefinition(value_id, "value-1"),),
    )
    assert system.get(value_id) == quarantine
    assert not system.get(value_id).is_active()
    assert system.conflicts == (ValueConflictDefinition(value_id, "value-1"),)
    with pytest.raises(ValueDomainError):
        system.apply_update(_admission(value_id=value_id), _event())


def _restorable_system(update_count: int = 2) -> ValueSystem:
    system = ValueSystem((_mutable_value(),))
    for index in range(update_count):
        event_id = f"restore-event-{index}"
        system.apply_update(
            _admission(
                event_id=event_id,
                event_sequence=index,
                evidence_ref=f"restore-{index}",
            ),
            _event(event_id, index),
        )
    return system


def test_restore_reconstructs_revision_history_and_exact_ledger_without_replay() -> None:
    source = _restorable_system()
    restored = ValueSystem.restore(
        values=source.value_map,
        conflicts=source.conflicts,
        histories=source.histories,
        evidence_ledgers=source.evidence_ledgers,
        evidence_ledger_digests=source.evidence_ledger_digests,
    )
    assert restored.values == source.values
    assert restored.histories == source.histories
    assert restored.evidence_ledgers == source.evidence_ledgers
    assert restored.evidence_ledger_digests == source.evidence_ledger_digests
    replay = restored.apply_update(
        _admission(
            event_id="restore-replay",
            event_sequence=20,
            evidence_ref="restore-0",
        ),
        _event("restore-replay", 20),
    )
    assert replay.status is ValueMutationStatus.IDEMPOTENT
    fresh = restored.apply_update(
        _admission(
            event_id="restore-new",
            event_sequence=21,
            evidence_ref="restore-new",
        ),
        _event("restore-new", 21),
    )
    assert fresh.status is ValueMutationStatus.APPLIED


def test_restore_accepts_inactive_values_and_requires_exact_key_sets() -> None:
    quarantine = _value(
        value_id="quarantine-1",
        origin=IdentityOrigin(
            OriginActor.INHERITED,
            OriginInputKind.LEGACY,
            ValueAdmissionStatus.UNCERTAIN,
        ),
        evidence_refs=(),
    )
    source = _restorable_system()
    values = {**source.value_map, "quarantine-1": quarantine}
    histories = {**source.histories, "quarantine-1": ValueRevisionHistory("quarantine-1")}
    ledgers = {**source.evidence_ledgers, "quarantine-1": ()}
    restored = ValueSystem.restore(
        values=values,
        conflicts=(),
        histories=histories,
        evidence_ledgers=ledgers,
    )
    assert not restored.get("quarantine-1").is_active()
    with pytest.raises(ValueDomainError):
        ValueSystem.restore(
            values=values,
            conflicts=(),
            histories={"value-1": histories["value-1"]},
            evidence_ledgers=ledgers,
        )


def test_restore_rejects_current_history_and_ledger_inconsistency() -> None:
    source = _restorable_system()
    current = source.get("value-1")
    bad_current = replace(current, strength=current.strength - 0.01)
    with pytest.raises(ValueDomainError):
        ValueSystem.restore(
            values={"value-1": bad_current},
            conflicts=(),
            histories=source.histories,
            evidence_ledgers=source.evidence_ledgers,
        )

    missing_current_ref = tuple(
        ref for ref in source.applied_evidence("value-1") if ref != current.evidence_refs[0]
    )
    with pytest.raises(ValueDomainError):
        ValueSystem.restore(
            values=source.value_map,
            conflicts=(),
            histories=source.histories,
            evidence_ledgers={"value-1": missing_current_ref},
        )

    long_source = _restorable_system(33)
    long_current = long_source.get("value-1")
    long_history = long_source.history("value-1")
    historical_ref = next(
        record.evidence_refs[0]
        for record in long_history.records
        if record.evidence_refs[0] not in long_current.evidence_refs
    )
    missing_historical_ref = tuple(
        ref for ref in long_source.applied_evidence("value-1") if ref != historical_ref
    )
    with pytest.raises(ValueDomainError):
        ValueSystem.restore(
            values=long_source.value_map,
            conflicts=(),
            histories=long_source.histories,
            evidence_ledgers={"value-1": missing_historical_ref},
        )
    valid = ValueSystem.restore(
        values=long_source.value_map,
        conflicts=(),
        histories=long_source.histories,
        evidence_ledgers=long_source.evidence_ledgers,
    )
    assert historical_ref in valid.applied_evidence("value-1")


def test_restore_rejects_surplus_ledger_ref_with_exact_digest_witness() -> None:
    source = _restorable_system(33)
    refs = tuple(sorted((*source.applied_evidence("value-1"), "surplus-ref")))
    with pytest.raises(ValueDomainError):
        ValueSystem.restore(
            values=source.value_map,
            conflicts=(),
            histories=source.histories,
            evidence_ledgers={"value-1": refs},
            evidence_ledger_digests=source.evidence_ledger_digests,
        )

    assert evidence_ledger_digest("value-1", refs) != source.evidence_ledger_digests[
        "value-1"
    ]


def test_value_system_snapshot_round_trip_is_replay_free() -> None:
    source = _restorable_system(33)
    restored = ValueSystem.restore_snapshot(source.snapshot())
    assert restored.values == source.values
    assert restored.histories == source.histories
    assert restored.evidence_ledgers == source.evidence_ledgers
    assert restored.evidence_ledger_digests == source.evidence_ledger_digests


def test_revision_history_rejects_broken_state_continuity() -> None:
    source = _restorable_system()
    first, second = source.history("value-1").records
    broken = replace(second, before_digest="0" * 64)
    with pytest.raises(ValueError):
        ValueRevisionHistory("value-1", records=(first, broken))


def test_admission_history_requires_genesis_and_ordinary_history_starts_at_zero() -> None:
    candidate = _value(evidence_refs=("admission-1",))
    system = ValueSystem()
    admission = _admission(
        reason=ValueMutationReason.SELF_ADMISSION,
        evidence_ref="admission-1",
    )
    result = system.admit_self_value(candidate, admission, _event())
    assert result.revision_record is not None
    with pytest.raises(ValueError):
        replace(result.revision_record, before_digest="0" * 64)

    after = _value(revision=2)
    ordinary = ValueRevisionRecord(
        value_id=after.value_id,
        from_revision=1,
        to_revision=2,
        before_digest="0" * 64,
        after_state_projection=after,
        after_digest=value_state_digest(after),
        operation=ValueRevisionOperation.UPDATE,
        origin_id=after.origin.origin_id,
        evidence_refs=after.evidence_refs,
        event_id="event-1",
        event_sequence=0,
        recorded_at=_event().recorded_at,
    )
    with pytest.raises(ValueError):
        ValueRevisionHistory("value-1", records=(ordinary,))

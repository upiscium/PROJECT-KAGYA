import hashlib
from dataclasses import replace

import pytest

from kagya.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
)
from kagya.identity.value_system import (
    ValueConflictDefinition,
    ValueEvidence,
    ValueProposal,
    ValueReason,
    ValueSeedDeclaration,
    ValueScope,
    ValueState,
    canonical_seed_payload,
    recompute_seed_contract_digest,
    validate_seed_contract_digest,
)


def _origin(*, system: bool = False) -> IdentityOrigin:
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
        event_id="event-1",
        event_sequence=0,
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


def test_system_seed_digest_must_match_its_declaration() -> None:
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
    with pytest.raises(ValueError):
        _value(
            origin=_origin(system=True),
            seed_contract_digest="0" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value_id", "other-value"),
        ("name", "different"),
        ("concept", "different concept"),
        ("scope", ValueScope.CONTEXT),
        ("context_ids", ("context-1",)),
        ("polarity", -1),
        ("strength", 0.7),
        ("confidence", 0.8),
        ("stability", 0.6),
        ("protectedness", 0.5),
        ("negotiability", 0.3),
        ("allowed_update_rate", 0.2),
    ],
)
def test_system_value_fields_must_match_seed_digest(field: str, value: object) -> None:
    seed = _seed()
    with pytest.raises(ValueError):
        _value(
            **{field: value},
            origin=_origin(system=True),
            seed_contract_digest=recompute_seed_contract_digest(seed),
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

import hashlib

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


def test_scope_and_active_read_semantics() -> None:
    value = _value()
    assert value.is_active()
    assert value.applies_to(None)
    assert not value.applies_to("context-1")
    contextual = _value(scope=ValueScope.CONTEXT, context_ids=("context-1",))
    assert contextual.applies_to("context-1")
    assert not contextual.applies_to(None)
    with pytest.raises(ValueError):
        contextual.applies_to("bad..id")


@pytest.mark.parametrize(
    "field", ["strength", "confidence", "stability", "protectedness", "negotiability"]
)
def test_scalar_bounds_are_strict(field: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _value(**{field: 2.0})
    with pytest.raises(TypeError):
        _value(**{field: 1})


def test_refs_conflicts_and_non_authoritative_types() -> None:
    with pytest.raises(ValueError):
        _value(evidence_refs=("evidence-2", "evidence-1"))
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
    value = _value()
    payload = b'kagya.identity.value-seed/v1\x00{"allowed_update_rate":0.1,"concept":"Protect the wellbeing of the subject.","confidence":0.9,"context_ids":[],"initial_strength":0.8,"name":"care","negotiability":0.2,"polarity":1,"protectedness":0.6,"scope":"subject","stability":0.7,"value_id":"value-1"}'
    expected = "081ba5fbfc1075cfa2246790e137080f9eb500e4ecf1e3f42b6c322ef641e146"
    assert canonical_seed_payload(value) == payload
    assert expected == hashlib.sha256(payload).hexdigest()
    assert recompute_seed_contract_digest(value) == expected
    assert (
        validate_seed_contract_digest(
            _value(origin=_origin(system=True), seed_contract_digest=expected)
        )
        == expected
    )
    assert recompute_seed_contract_digest(_value(name="different")) != expected


def test_system_seed_digest_must_match_its_declaration() -> None:
    with pytest.raises(ValueError):
        _value(
            origin=_origin(system=True),
            seed_contract_digest="0" * 64,
        )

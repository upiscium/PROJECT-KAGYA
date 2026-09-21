import pytest

from kagya.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
    recompute_origin_id,
    validate_origin_id,
)


def test_closed_enum_values() -> None:
    assert [member.value for member in OriginActor] == [
        "self", "user", "operator", "system", "external_source",
        "model_inference", "inherited", "unknown",
    ]
    assert {member.value for member in OriginInputKind} == {
        "internal_state", "request", "suggestion", "constraint", "feedback",
        "evidence", "config_seed", "legacy",
    }
    assert {member.value for member in ValueAdmissionStatus} == {
        "pending", "self_endorsed", "system_authorized", "rejected", "uncertain",
    }
    with pytest.raises(ValueError):
        OriginActor("future_actor")


def test_admission_rules_and_required_self_evidence() -> None:
    self_origin = IdentityOrigin(
        OriginActor.SELF, OriginInputKind.INTERNAL_STATE,
        ValueAdmissionStatus.SELF_ENDORSED, event_id="evt-1", event_sequence=0,
    )
    system_origin = IdentityOrigin(
        OriginActor.SYSTEM, OriginInputKind.CONFIG_SEED,
        ValueAdmissionStatus.SYSTEM_AUTHORIZED,
    )
    assert self_origin.origin_id and system_origin.origin_id
    pending_self_origin = IdentityOrigin(
        OriginActor.SELF,
        OriginInputKind.INTERNAL_STATE,
        ValueAdmissionStatus.PENDING,
        event_id="evt-1",
        event_sequence=0,
    )
    assert pending_self_origin.origin_id
    with pytest.raises(ValueError):
        IdentityOrigin(
            OriginActor.SELF,
            OriginInputKind.INTERNAL_STATE,
            ValueAdmissionStatus.SELF_ENDORSED,
        )
    with pytest.raises(ValueError):
        IdentityOrigin(OriginActor.USER, OriginInputKind.REQUEST,
                       ValueAdmissionStatus.SELF_ENDORSED)
    with pytest.raises(ValueError):
        IdentityOrigin(OriginActor.SYSTEM, OriginInputKind.CONSTRAINT,
                       ValueAdmissionStatus.SYSTEM_AUTHORIZED)
    with pytest.raises(ValueError):
        IdentityOrigin(OriginActor.UNKNOWN, OriginInputKind.REQUEST,
                       ValueAdmissionStatus.SELF_ENDORSED)


@pytest.mark.parametrize(
    "actor",
    [
        OriginActor.USER,
        OriginActor.OPERATOR,
        OriginActor.EXTERNAL_SOURCE,
        OriginActor.MODEL_INFERENCE,
    ],
)
def test_external_actors_cannot_construct_active_admissions(actor: OriginActor) -> None:
    with pytest.raises(ValueError):
        IdentityOrigin(actor, OriginInputKind.EVIDENCE, ValueAdmissionStatus.SELF_ENDORSED)
    with pytest.raises(ValueError):
        IdentityOrigin(actor, OriginInputKind.EVIDENCE, ValueAdmissionStatus.SYSTEM_AUTHORIZED)


@pytest.mark.parametrize("actor", [OriginActor.OPERATOR, OriginActor.SYSTEM])
def test_constraints_cannot_construct_active_admissions(actor: OriginActor) -> None:
    with pytest.raises(ValueError):
        IdentityOrigin(actor, OriginInputKind.CONSTRAINT, ValueAdmissionStatus.SELF_ENDORSED)
    with pytest.raises(ValueError):
        IdentityOrigin(actor, OriginInputKind.CONSTRAINT, ValueAdmissionStatus.SYSTEM_AUTHORIZED)


@pytest.mark.parametrize("kwargs", [{"event_id": "event-1"}, {"event_sequence": 0}])
def test_self_endorsement_requires_event_id_and_sequence(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IdentityOrigin(
            OriginActor.SELF,
            OriginInputKind.INTERNAL_STATE,
            ValueAdmissionStatus.SELF_ENDORSED,
            **kwargs,
        )


def test_origin_is_deterministic_and_admission_independent() -> None:
    kwargs = dict(actor=OriginActor.USER, input_kind=OriginInputKind.EVIDENCE,
                  source_ref="source:1", event_id="event:1", context_id="ctx:1",
                  event_sequence=4, confidence=0.5)
    pending = IdentityOrigin(admission=ValueAdmissionStatus.PENDING, **kwargs)
    uncertain = IdentityOrigin(admission=ValueAdmissionStatus.UNCERTAIN, **kwargs)
    assert pending.origin_id == uncertain.origin_id == recompute_origin_id(pending)
    assert validate_origin_id(pending) == pending.origin_id


@pytest.mark.parametrize("kwargs", [
    {"source_ref": "bad ref"}, {"event_id": ".."}, {"context_id": ""},
    {"event_sequence": -1}, {"event_sequence": True}, {"confidence": float("nan")},
    {"confidence": 1.1}, {"confidence": 1},
])
def test_malformed_optional_fields_are_rejected(kwargs: dict[str, object]) -> None:
    base = dict(actor=OriginActor.USER, input_kind=OriginInputKind.REQUEST,
                admission=ValueAdmissionStatus.PENDING)
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        IdentityOrigin(**base)

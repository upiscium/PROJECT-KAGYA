from dataclasses import replace

import pytest

from kagya.identity import (
    EndorsementStatus,
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    identity_origin_from_json,
    legacy_identity_origin,
    new_identity_origin,
)


def test_external_request_requires_explicit_subject_endorsement() -> None:
    request = new_identity_origin(
        OriginActor.USER,
        OriginInputKind.REQUEST,
        source_ref="context:one",
        event_id="event-1",
        event_sequence=1,
    )

    assert request.endorsement == EndorsementStatus.PENDING
    with pytest.raises(ValueError, match="explicit subject endorsement"):
        replace(request, endorsement=EndorsementStatus.ENDORSED)

    endorsed = request.endorse(
        "goal_adoption", event_id="event-2", event_sequence=2
    )
    assert endorsed.actor == OriginActor.USER
    assert endorsed.endorsement == EndorsementStatus.ENDORSED
    assert endorsed.endorsed_by_event_id == "event-2"


def test_constraints_are_distinct_from_subject_endorsement() -> None:
    constraint = new_identity_origin(
        OriginActor.OPERATOR,
        OriginInputKind.CONSTRAINT,
        source_ref="policy:local",
    )

    assert constraint.endorsement == EndorsementStatus.IMPOSED
    assert constraint.actor != OriginActor.SELF


def test_legacy_origin_is_uncertain_and_round_trips() -> None:
    origin = legacy_identity_origin("legacy_snapshot")

    restored = identity_origin_from_json(origin.to_json())

    assert restored == origin
    assert restored.actor == OriginActor.INHERITED
    assert restored.endorsement == EndorsementStatus.UNCERTAIN


def test_origin_rejects_free_text_references() -> None:
    with pytest.raises(ValueError, match="opaque safe reference"):
        IdentityOrigin(
            origin_id="origin-1",
            actor=OriginActor.USER,
            input_kind=OriginInputKind.REQUEST,
            endorsement=EndorsementStatus.PENDING,
            source_ref="private user sentence with spaces",
            event_id=None,
            event_sequence=None,
            context_id=None,
            confidence=1.0,
            created_at="2026-01-01T00:00:00+00:00",
        )

import pytest

from kagya.cognition import AppraisalResult
from kagya.experience import build_chat_experience
from kagya.identity import OriginActor, OriginInputKind, new_identity_origin
from kagya.relationship import PerceivedAttribute, RelationshipStore


def _experience(
    identifier: str,
    *,
    interlocutor: str = "person-1",
    valence: float = 0.4,
    threat: float = 0.0,
    certainty: float = 0.8,
):
    return build_chat_experience(
        source_event_id=f"event-{identifier}",
        source_event_sequence=1,
        episode_id=identifier,
        identity_origin=new_identity_origin(
            OriginActor.USER,
            OriginInputKind.OBSERVATION,
            source_ref=f"context:{identifier}",
        ),
        context_id=f"context-{identifier}",
        interlocutor_ids=(interlocutor,),
        appraisal=AppraisalResult(
            novelty=0.2,
            goal_progress=0.2,
            threat=threat,
            controllability=0.5,
            certainty=certainty,
            social_relevance=0.8,
            effort_cost=0.1,
            novelty_valid=True,
            reasons=("test_evidence",),
        ),
        valence=valence,
        arousal=0.2,
        prediction_error=0.1,
        value_revision_refs={},
        active_goal_refs=("goal:shared",),
        self_model_revision=0,
    )


def test_relationship_requires_repeated_evidence_and_bounds_independent_axes() -> None:
    store = RelationshipStore()

    first = store.observe_experience(_experience("one"))[0]
    second = store.observe_experience(_experience("two"))[0]

    assert first.axes.trust == 0.5
    assert first.axes.familiarity == 0.0
    assert second.axes.trust == pytest.approx(0.54)
    assert second.axes.familiarity == pytest.approx(0.05)
    assert second.axes.closeness == pytest.approx(0.14)
    assert second.axes.caution == pytest.approx(0.22)
    assert (
        max(
            abs(getattr(second.axes, name) - getattr(first.axes, name))
            for name in ("trust", "familiarity", "closeness", "caution")
        )
        <= store.MAX_AXIS_UPDATE
    )
    assert second.shared_experience_refs == (
        f"experience:{first.evidence[0].experience_id}",
        f"experience:{second.evidence[-1].experience_id}",
    )
    assert second.goal_refs == ("goal:shared",)


def test_relationship_mapping_rejects_ambiguous_merge_and_supports_split() -> None:
    store = RelationshipStore()
    first = store.ensure_interlocutor("person-1")
    second = store.ensure_interlocutor("person-2")

    with pytest.raises(ValueError, match="two independent"):
        store.attach_alias(
            first.relationship_id, "alias", evidence_refs=("evidence:1",)
        )
    with pytest.raises(ValueError, match="another relationship"):
        store.attach_alias(
            first.relationship_id,
            "person-2",
            evidence_refs=("evidence:1", "evidence:2"),
        )

    merged = store.attach_alias(
        first.relationship_id,
        "alias",
        evidence_refs=("evidence:1", "evidence:2"),
    )
    split = store.split_alias(
        merged.relationship_id,
        "alias",
        reason="operator_identity_correction",
        evidence_refs=("evidence:3",),
    )

    assert store.for_interlocutor("person-1").relationship_id == first.relationship_id
    assert split.relationship_id not in {first.relationship_id, second.relationship_id}
    assert store.for_interlocutor("alias") == split


def test_other_person_values_and_beliefs_are_separate_and_round_trip() -> None:
    store = RelationshipStore()
    state = store.ensure_interlocutor("person-1")
    corrected = store.correct(
        state.relationship_id,
        reason="operator_review",
        evidence_refs=("experience:reviewed",),
        perceived_role=PerceivedAttribute(
            "collaborator", 0.8, ("experience:reviewed",)
        ),
        other_values={
            "privacy": PerceivedAttribute("important", 0.7, ("experience:reviewed",))
        },
        other_beliefs={
            "schedule": PerceivedAttribute("flexible", 0.6, ("experience:reviewed",))
        },
    )

    restored = RelationshipStore()
    restored.restore(store.to_json())
    value = restored.get(corrected.relationship_id)

    assert value.other_values["privacy"].value == "important"
    assert value.other_beliefs["schedule"].value == "flexible"
    assert value.perceived_role.value == "collaborator"
    assert value.revisions[-1].evidence_refs == ("experience:reviewed",)


def test_commitment_outcomes_preserve_breach_and_repair_continuity() -> None:
    store = RelationshipStore()
    state = store.link_commitment("person-1", "commitment:one")

    breached = store.transition_commitment(
        "commitment:one",
        status="breached",
        evidence_ref="commitment:one:breached",
    )[0]
    fulfilled = store.transition_commitment(
        "commitment:one",
        status="fulfilled",
        evidence_ref="commitment:one:fulfilled",
    )[0]

    assert "commitment:one" in state.unresolved_matter_refs
    assert breached.conflict_refs == ("commitment:one:breached",)
    assert breached.unresolved_matter_refs == ("commitment:one:breached",)
    assert fulfilled.unresolved_matter_refs == ()
    assert fulfilled.repair_refs == ("commitment:one:fulfilled",)

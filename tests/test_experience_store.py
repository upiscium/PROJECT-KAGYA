import json

import pytest

from kagya.cognition import AppraisalResult
from kagya.experience import (
    ExperienceAppraisal,
    ExperienceStore,
    build_chat_experience,
)
from kagya.identity import OriginActor, OriginInputKind, new_identity_origin


def test_chat_experience_separates_observation_interpretation_and_private_content() -> (
    None
):
    record = _experience(event_id="event-1", episode_id="episode-1")

    payload = record.to_json()
    serialized = json.dumps(payload)

    assert record.external_observation_refs == ("episode:episode-1:input",)
    assert record.subject_action_refs == ("episode:episode-1:response",)
    assert record.interpretation_codes == ("novelty_measured",)
    assert record.result_refs == {"memory": ("episode:episode-1",)}
    assert "private request text" not in serialized
    assert "generated response text" not in serialized
    assert "hidden_thought" not in serialized


def test_store_deduplicates_source_event_and_observation() -> None:
    store = ExperienceStore()
    first = store.integrate(_experience(event_id="event-1", episode_id="episode-1"))
    duplicate_event = store.integrate(
        _experience(event_id="event-1", episode_id="episode-2")
    )
    duplicate_observation = store.integrate(
        _experience(event_id=None, episode_id="episode-1")
    )

    assert duplicate_event != first
    assert duplicate_observation == first
    assert len(store.list_records()) == 2


def test_store_allows_same_event_observation_semantics_in_another_context() -> None:
    store = ExperienceStore()
    first = store.integrate(_experience(event_id="event-1", episode_id="episode-1"))
    other = store.integrate(
        _experience(
            event_id="event-1", episode_id="episode-1", context_id="context-other"
        )
    )

    assert other.experience_id != first.experience_id
    assert len(store.list_records()) == 2


def test_context_and_appraisal_change_subjective_meaning_and_salience() -> None:
    familiar = _experience(
        event_id="event-familiar",
        episode_id="episode-familiar",
        novelty=0.1,
        arousal=0.1,
        social_relevance=0.0,
        interlocutor_ids=(),
    )
    interpersonal = _experience(
        event_id="event-interpersonal",
        episode_id="episode-interpersonal",
        novelty=0.9,
        arousal=0.8,
        social_relevance=0.9,
        interlocutor_ids=("person-1",),
    )

    assert familiar.subjective_salience < interpersonal.subjective_salience
    assert familiar.self_relevance < interpersonal.self_relevance
    assert "interpersonal_context" not in familiar.situation_codes
    assert "interpersonal_context" in interpersonal.situation_codes


def test_reassessment_and_result_links_are_revision_controlled() -> None:
    store = ExperienceStore()
    original = store.integrate(_experience(event_id="event-1", episode_id="episode-1"))
    reassessed = store.revise_appraisal(
        original.experience_id,
        appraisal=ExperienceAppraisal(
            valence=-0.4,
            arousal=0.8,
            novelty=0.9,
            novelty_valid=True,
            goal_progress=-0.5,
            threat=0.7,
            controllability=0.2,
            certainty=0.9,
            social_relevance=0.8,
            effort_cost=0.3,
            reason_codes=("later_evidence",),
        ),
        reason_code="later_evidence",
        evidence_refs=("memory:evidence-1",),
        event_id="event-2",
        event_sequence=2,
    )
    linked = store.link_result(
        original.experience_id,
        kind="goal",
        reference="goal:follow-up",
        evidence_refs=("event:event-3",),
        event_id="event-3",
        event_sequence=3,
    )

    assert reassessed.revision == 1
    assert reassessed.subjective_salience > original.subjective_salience
    assert linked.revision == 2
    assert linked.result_refs["goal"] == ("goal:follow-up",)
    assert linked.revisions[-1].changed_fields == ("result_refs",)


def test_store_round_trip_preserves_indexes_and_rejects_unknown_version() -> None:
    store = ExperienceStore()
    original = store.integrate(_experience(event_id="event-1", episode_id="episode-1"))
    payload = json.loads(json.dumps(store.to_json()))
    restored = ExperienceStore()
    restored.restore(payload)

    duplicate = restored.integrate(
        _experience(event_id="event-1", episode_id="episode-1")
    )

    assert duplicate == original
    assert restored.to_json() == payload
    with pytest.raises(ValueError, match="Unsupported experience store"):
        restored.restore({"schema_version": 2, "records": []})


def _experience(
    *,
    event_id: str | None,
    episode_id: str,
    novelty: float = 0.4,
    arousal: float = 0.3,
    social_relevance: float = 0.2,
    interlocutor_ids: tuple[str, ...] = (),
    context_id: str = "context-one",
):
    return build_chat_experience(
        source_event_id=event_id,
        source_event_sequence=1 if event_id is not None else None,
        episode_id=episode_id,
        identity_origin=new_identity_origin(
            OriginActor.USER,
            OriginInputKind.OBSERVATION,
            source_ref="context:one",
            event_id=event_id,
        ),
        context_id=context_id,
        interlocutor_ids=interlocutor_ids,
        appraisal=AppraisalResult(
            novelty=novelty,
            goal_progress=0.0,
            threat=0.0,
            controllability=0.5,
            certainty=0.8,
            social_relevance=social_relevance,
            effort_cost=0.2,
            novelty_valid=True,
            reasons=("novelty_measured",),
        ),
        valence=0.1,
        arousal=arousal,
        prediction_error=0.4,
        value_revision_refs={"care": 1},
        active_goal_refs=(),
        self_model_revision=0,
    )

"""Focused tests for the pure R12 Experience evidence boundary."""

from dataclasses import fields, replace
from datetime import UTC, datetime

import pytest

from kagya.body import EmotionUpdateReasonCode, ValenceContributions
from kagya.cognition import AppraisalReasonCode, LossInvalidReason
from kagya.experience import (
    ExperienceAppraisalEvidence,
    ExperienceEmotionContributions,
    ExperienceEmotionProjection,
    ExperienceLifecycle,
    ExperienceMeasurementEvidence,
    EXPERIENCE_MAX_REVISION,
    ExperienceRecord,
    calculate_subjective_salience,
)


MODEL_KEY = "model." + "a" * 64
CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_invalid_measurement_stays_invalid_instead_of_becoming_zero() -> None:
    measurement = ExperienceMeasurementEvidence(
        model_key=MODEL_KEY,
        valid=False,
        invalid_reason=LossInvalidReason.EMPTY_TARGET,
    )
    assert measurement.calibrated_novelty is None
    assert measurement.invalid_reason is LossInvalidReason.EMPTY_TARGET
    with pytest.raises(ValueError):
        ExperienceMeasurementEvidence(
            model_key=MODEL_KEY,
            valid=False,
            invalid_reason=LossInvalidReason.EMPTY_TARGET,
            calibrated_novelty=0.0,
        )


def test_salience_golden_fixtures_cover_valid_and_invalid_novelty() -> None:
    pre = ExperienceEmotionProjection(valence=0.0, arousal=0.0)
    post = ExperienceEmotionProjection(valence=0.5, arousal=0.25)
    valid = ExperienceMeasurementEvidence(MODEL_KEY, True, calibrated_novelty=0.25)
    invalid = ExperienceMeasurementEvidence(
        MODEL_KEY, False, invalid_reason=LossInvalidReason.PROVIDER_ERROR
    )
    assert calculate_subjective_salience(valid, pre, post) == 0.25
    assert calculate_subjective_salience(invalid, pre, post) == 0.25


def test_r10_optional_appraisal_and_emotion_bounds_are_preserved() -> None:
    appraisal = ExperienceAppraisalEvidence(
        novelty=None,
        novelty_valid=False,
        reason_codes=(AppraisalReasonCode.NOVELTY_INVALID,),
    )
    assert appraisal.goal_progress is None
    assert appraisal.threat is None
    assert appraisal.novelty is None
    with pytest.raises(ValueError):
        ValenceContributions(goal_progress=0.8)
    assert ExperienceEmotionContributions().valence.goal_progress == 0.0


def test_experience_record_is_immutable_bounded_and_rejects_raw_content_fields() -> None:
    measurement = ExperienceMeasurementEvidence(MODEL_KEY, True, calibrated_novelty=0.25)
    pre = ExperienceEmotionProjection(0.0, 0.0)
    post = ExperienceEmotionProjection(0.5, 0.25)
    record = ExperienceRecord(
        experience_id="experience:1",
        revision=0,
        lifecycle=ExperienceLifecycle.ACTIVE,
        source_event_id="event:1",
        source_event_sequence=1,
        source_episode_id="episode:1",
        context_id="context:1",
        measurement=measurement,
        appraisal=ExperienceAppraisalEvidence(
            novelty=0.25,
            novelty_valid=True,
            reason_codes=(AppraisalReasonCode.NOVELTY_MEASURED,),
        ),
        pre_appraisal_emotion=pre,
        temporal_update_reasons=(EmotionUpdateReasonCode.TIMELINE_INITIALIZED,),
        post_appraisal_emotion=post,
        emotion_contributions=ExperienceEmotionContributions(),
        emotion_update_reasons=(EmotionUpdateReasonCode.APPRAISAL_APPLIED,),
        subjective_salience=0.25,
        created_at=CREATED_AT,
    )
    assert record.subjective_salience == 0.25
    with pytest.raises(AttributeError):
        record.context_id = "context:2"  # type: ignore[misc]
    names = {item.name for item in fields(ExperienceRecord)}
    assert not names.intersection(
        {"raw_text", "prompt", "response", "transcript", "hidden_reasoning", "narrative"}
    )
    with pytest.raises(ValueError):
        ExperienceRecord(
            experience_id="experience:1",
            revision=0,
            lifecycle=ExperienceLifecycle.ACTIVE,
            source_event_id="event:1",
            source_event_sequence=1,
            source_episode_id="episode:1",
            context_id=None,
            measurement=measurement,
            appraisal=record.appraisal,
            pre_appraisal_emotion=pre,
            temporal_update_reasons=(),
            post_appraisal_emotion=post,
            emotion_contributions=ExperienceEmotionContributions(),
            emotion_update_reasons=(),
            subjective_salience=0.0,
            created_at=CREATED_AT,
        )
    with pytest.raises(ValueError):
        replace(record, source_event_sequence=2**63)
    with pytest.raises(ValueError):
        replace(record, revision=1)
    with pytest.raises(ValueError):
        replace(record, revision=EXPERIENCE_MAX_REVISION + 1)

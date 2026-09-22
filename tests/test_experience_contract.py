"""Focused tests for the pure R12 Experience evidence boundary."""

from dataclasses import fields, replace
from datetime import UTC, datetime

import pytest

from kagya.body import (
    ArousalContributions,
    EmotionState,
    EmotionUpdate,
    EmotionUpdateReasonCode,
    ValenceContributions,
)
from kagya.cognition import (
    AppraisalReasonCode,
    AppraisalResult,
    LossInvalidReason,
    LossMeasurement,
)
from kagya.experience import (
    ExperienceAppraisalEvidence,
    ExperienceAppraisalReasonCode,
    ExperienceArousalContributions,
    ExperienceEmotionContributions,
    ExperienceEmotionUpdateReasonCode,
    ExperienceEmotionProjection,
    ExperienceLifecycle,
    ExperienceMeasurementEvidence,
    ExperienceMeasurementInvalidReason,
    EXPERIENCE_MAX_REVISION,
    ExperienceRecord,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    ExperienceRevisionRecord,
    ExperienceValenceContributions,
    calculate_subjective_salience,
    experience_revision_digest,
)


MODEL_KEY = "model." + "a" * 64
CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_invalid_measurement_stays_invalid_instead_of_becoming_zero() -> None:
    measurement = ExperienceMeasurementEvidence(
        model_key=MODEL_KEY,
        valid=False,
        invalid_reason=ExperienceMeasurementInvalidReason.EMPTY_TARGET,
    )
    assert measurement.calibrated_novelty is None
    assert measurement.invalid_reason is ExperienceMeasurementInvalidReason.EMPTY_TARGET
    with pytest.raises(ValueError):
        ExperienceMeasurementEvidence(
            model_key=MODEL_KEY,
            valid=False,
            invalid_reason=ExperienceMeasurementInvalidReason.EMPTY_TARGET,
            calibrated_novelty=0.0,
        )


def test_salience_golden_fixtures_cover_valid_and_invalid_novelty() -> None:
    pre = ExperienceEmotionProjection(valence=0.0, arousal=0.0)
    post = ExperienceEmotionProjection(valence=0.5, arousal=0.25)
    valid = ExperienceMeasurementEvidence(MODEL_KEY, True, calibrated_novelty=0.25)
    invalid = ExperienceMeasurementEvidence(
        MODEL_KEY,
        False,
        invalid_reason=ExperienceMeasurementInvalidReason.PROVIDER_ERROR,
    )
    assert calculate_subjective_salience(valid, pre, post) == 0.25
    assert calculate_subjective_salience(invalid, pre, post) == 0.25


def test_r10_optional_appraisal_and_emotion_bounds_are_preserved() -> None:
    appraisal = ExperienceAppraisalEvidence(
        novelty=None,
        novelty_valid=False,
        reason_codes=(ExperienceAppraisalReasonCode.NOVELTY_INVALID,),
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
    genesis = ExperienceRevisionRecord(
        "experience:1",
        0,
        ExperienceRevisionOperation.CREATE,
        ExperienceRevisionReason.CREATION,
        CREATED_AT,
        event_id="event:1",
        event_sequence=1,
        evidence_refs=("event:1",),
    )
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
            reason_codes=(ExperienceAppraisalReasonCode.NOVELTY_MEASURED,),
        ),
        pre_appraisal_emotion=pre,
        temporal_update_reasons=(ExperienceEmotionUpdateReasonCode.TIMELINE_INITIALIZED,),
        post_appraisal_emotion=post,
        emotion_contributions=ExperienceEmotionContributions(),
        emotion_update_reasons=(ExperienceEmotionUpdateReasonCode.APPRAISAL_APPLIED,),
        subjective_salience=0.25,
        created_at=CREATED_AT,
        revision_history=(genesis,),
    )
    assert record.subjective_salience == 0.25
    with pytest.raises(AttributeError):
        record.context_id = "context:2"  # type: ignore[misc]
    names = {item.name for item in fields(ExperienceRecord)}
    assert not names.intersection(
        {"raw_text", "prompt", "response", "transcript", "hidden_reasoning", "narrative"}
    )
    with pytest.raises((TypeError, ValueError)):
        replace(record, context_id=None)
    with pytest.raises(ValueError):
        replace(record, source_event_sequence=2**63)
    with pytest.raises(ValueError):
        replace(record, source_event_sequence=0)
    with pytest.raises(ValueError):
        replace(record, source_event_sequence=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(record, revision=1, revision_history=())
    with pytest.raises(ValueError):
        replace(record, history_anchor_digest="0" * 64)
    with pytest.raises(ValueError):
        replace(record, revision=EXPERIENCE_MAX_REVISION + 1)


def test_r10_adapters_map_to_frozen_r12_owned_types() -> None:
    measurement = ExperienceMeasurementEvidence.from_loss_measurement(
        LossMeasurement(
            MODEL_KEY,
            None,
            False,
            LossInvalidReason.EMPTY_TARGET,
            None,
        )
    )
    assert isinstance(measurement.invalid_reason, ExperienceMeasurementInvalidReason)

    appraisal = ExperienceAppraisalEvidence.from_result(
        AppraisalResult(
            novelty=None,
            novelty_valid=False,
            reasons=(AppraisalReasonCode.NOVELTY_INVALID,),
        )
    )
    assert appraisal.reasons == (ExperienceAppraisalReasonCode.NOVELTY_INVALID,)

    update = EmotionUpdate(
        state=EmotionState(),
        valence_contributions=ValenceContributions(),
        arousal_contributions=ArousalContributions(),
        reasons=(EmotionUpdateReasonCode.NO_MATERIAL_APPRAISAL,),
    )
    contributions = ExperienceEmotionContributions.from_update(update)
    assert isinstance(contributions.valence, ExperienceValenceContributions)
    assert isinstance(contributions.arousal, ExperienceArousalContributions)
    with pytest.raises(TypeError):
        ExperienceMeasurementEvidence(
            MODEL_KEY,
            False,
            invalid_reason=LossInvalidReason.EMPTY_TARGET,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        ExperienceAppraisalEvidence(
            None,
            False,
            reason_codes=(AppraisalReasonCode.NOVELTY_INVALID,),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        ExperienceEmotionContributions(ValenceContributions(), ArousalContributions())  # type: ignore[arg-type]


def test_experience_revision_evidence_and_event_binding_are_strict() -> None:
    genesis = ExperienceRevisionRecord(
        "experience:1",
        0,
        ExperienceRevisionOperation.CREATE,
        ExperienceRevisionReason.CREATION,
        CREATED_AT,
        evidence_refs=("evidence:0",),
        event_id="event:1",
        event_sequence=1,
    )
    corrected = ExperienceRevisionRecord(
        "experience:1",
        1,
        ExperienceRevisionOperation.CORRECT,
        ExperienceRevisionReason.CORRECTION,
        CREATED_AT,
        evidence_refs=("evidence:1",),
        previous_revision_digest=genesis.record_digest,
        event_id="event:2",
        event_sequence=2,
    )
    assert experience_revision_digest(corrected) == corrected.record_digest
    assert corrected.record_digest != genesis.record_digest
    with pytest.raises(ValueError):
        ExperienceRevisionRecord(
            "experience:1",
            0,
            ExperienceRevisionOperation.CREATE,
            ExperienceRevisionReason.CREATION,
            CREATED_AT,
            event_id="event:1",
            event_sequence=1,
        )
    with pytest.raises(ValueError):
        ExperienceRevisionRecord(
            "experience:1",
            0,
            ExperienceRevisionOperation.CREATE,
            ExperienceRevisionReason.CREATION,
            CREATED_AT,
            evidence_refs=("evidence:0",),
            event_id="event:1",
            event_sequence=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        ExperienceRevisionRecord(
            "experience:1",
            0,
            ExperienceRevisionOperation.CREATE,
            ExperienceRevisionReason.CREATION,
            CREATED_AT,
            evidence_refs=("evidence:0",),
            event_id="event:1",
            event_sequence=0,
        )
    with pytest.raises(ValueError):
        ExperienceRevisionRecord(
            "experience:1",
            1,
            ExperienceRevisionOperation.CORRECT,
            ExperienceRevisionReason.CORRECTION,
            CREATED_AT,
            evidence_refs=("evidence:1",),
            event_id="event:2",
            event_sequence=2,
        )
    with pytest.raises(ValueError):
        ExperienceRevisionRecord(
            "experience:1",
            1,
            ExperienceRevisionOperation.CORRECT,
            ExperienceRevisionReason.CORRECTION,
            CREATED_AT,
            evidence_refs=("evidence:1",),
            previous_revision_digest=genesis.record_digest,
            event_id="event:2",
            event_sequence=True,  # type: ignore[arg-type]
        )
    with pytest.raises((TypeError, ValueError)):
        ExperienceRevisionRecord(
            "experience:1",
            1,
            ExperienceRevisionOperation.CORRECT,
            ExperienceRevisionReason.CORRECTION,
            CREATED_AT,
            evidence_refs=("evidence:1",),
            previous_revision_digest=genesis.record_digest,
            event_id=None,  # type: ignore[arg-type]
            event_sequence=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        ExperienceRevisionRecord(
            "experience:1",
            1,
            ExperienceRevisionOperation.CORRECT,
            ExperienceRevisionReason.REASSESSMENT,
            CREATED_AT,
            evidence_refs=("evidence:1",),
            previous_revision_digest=genesis.record_digest,
            event_id="event:2",
            event_sequence=2,
        )

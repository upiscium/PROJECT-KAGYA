from dataclasses import FrozenInstanceError
import math

import pytest

from kagya.cognition import (
    AppraisalReasonCode,
    AppraisalResult,
    AppraisalSignals,
    CognitiveAppraiser,
    LossInvalidReason,
    LossMeasurement,
)


MODEL_KEY = "model." + "a" * 64


def measurement(novelty: float | None = 0.75) -> LossMeasurement:
    if novelty is None:
        return LossMeasurement(
            MODEL_KEY, None, False, LossInvalidReason.PROVIDER_ERROR, None
        )
    return LossMeasurement(MODEL_KEY, 1.0, True, None, novelty)


def test_signals_are_immutable_bounded_and_keep_evidence_optional() -> None:
    signals = AppraisalSignals()
    assert signals == AppraisalSignals(None, None, None, None, None, None)
    assert all(
        getattr(signals, name) is None
        for name in (
            "goal_progress",
            "threat",
            "controllability",
            "certainty",
            "social_relevance",
            "effort_cost",
        )
    )
    with pytest.raises(FrozenInstanceError):
        signals.goal_progress = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "field, value",
    [
        ("goal_progress", -1.00001),
        ("goal_progress", 1.00001),
        ("threat", -0.00001),
        ("threat", 1.00001),
        ("controllability", math.inf),
        ("certainty", math.nan),
        ("social_relevance", True),
        ("effort_cost", "0.5"),
    ],
)
def test_signal_bounds_reject_invalid_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AppraisalSignals(**{field: value})  # type: ignore[arg-type]


def test_signals_retain_all_explicit_evidence() -> None:
    signals = AppraisalSignals(
        goal_progress=-0.5,
        threat=0.8,
        controllability=0.2,
        certainty=1.0,
        social_relevance=0.4,
        effort_cost=0.9,
    )

    result = CognitiveAppraiser().appraise(measurement(), signals)

    assert result.goal_progress == -0.5
    assert result.threat == 0.8
    assert result.controllability == 0.2
    assert result.certainty == 1.0
    assert result.social_relevance == 0.4
    assert result.effort_cost == 0.9
    assert result.reasons == (
        AppraisalReasonCode.NOVELTY_MEASURED,
        AppraisalReasonCode.GOAL_SETBACK,
        AppraisalReasonCode.THREAT,
    )


def test_reason_codes_are_closed_and_result_is_immutable() -> None:
    with pytest.raises((TypeError, ValueError)):
        AppraisalResult(
            0.5,
            True,
            reasons=("secret reason",),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        AppraisalResult(
            0.5,
            True,
            reasons=(AppraisalReasonCode.THREAT, AppraisalReasonCode.THREAT),
        )
    result = CognitiveAppraiser().appraise(measurement(), AppraisalSignals())
    with pytest.raises(FrozenInstanceError):
        result.novelty = 0.2  # type: ignore[misc]


def test_invalid_novelty_is_preserved_as_invalid_without_guessing() -> None:
    result = CognitiveAppraiser().appraise(measurement(None), AppraisalSignals())

    assert result.novelty is None
    assert not result.novelty_valid
    assert result.reasons == (AppraisalReasonCode.NOVELTY_INVALID,)
    assert all(
        getattr(result, name) is None
        for name in (
            "goal_progress",
            "threat",
            "controllability",
            "certainty",
            "social_relevance",
            "effort_cost",
        )
    )


def test_measured_zero_novelty_remains_distinct_from_invalid_novelty() -> None:
    measured = CognitiveAppraiser().appraise(
        LossMeasurement(MODEL_KEY, 1.0, True, None, 0.0), AppraisalSignals()
    )
    invalid = CognitiveAppraiser().appraise(measurement(None), AppraisalSignals())

    assert measured.novelty == 0.0
    assert measured.novelty_valid is True
    assert measured.reasons == (AppraisalReasonCode.NOVELTY_MEASURED,)
    assert invalid.novelty is None
    assert invalid.novelty_valid is False
    assert invalid.reasons == (AppraisalReasonCode.NOVELTY_INVALID,)


def test_appraisal_is_pure_deterministic_and_does_not_consume_private_text() -> None:
    signals = AppraisalSignals(threat=None)
    appraiser = CognitiveAppraiser()
    first = appraiser.appraise(measurement(), signals)
    second = appraiser.appraise(measurement(), signals)

    assert first == second
    assert "private provider exception sentinel" not in repr(first)
    assert appraiser.appraise(measurement(), AppraisalSignals(goal_progress=-0.4)) != first


@pytest.mark.parametrize(
    ("threat", "has_threat_reason"),
    [(None, False), (0.0, False), (0.0001, True)],
)
def test_threat_reason_requires_positive_threat_evidence(
    threat: float | None, has_threat_reason: bool
) -> None:
    result = CognitiveAppraiser().appraise(
        measurement(), AppraisalSignals(threat=threat)
    )

    assert (AppraisalReasonCode.THREAT in result.reasons) is has_threat_reason


def test_same_novelty_with_explicit_goal_and_threat_evidence_differs() -> None:
    appraiser = CognitiveAppraiser()
    neutral = appraiser.appraise(measurement(0.75), AppraisalSignals())
    evidence = appraiser.appraise(
        measurement(0.75), AppraisalSignals(goal_progress=0.5, threat=0.7)
    )

    assert neutral.novelty == evidence.novelty
    assert neutral != evidence
    assert evidence.reasons == (
        AppraisalReasonCode.NOVELTY_MEASURED,
        AppraisalReasonCode.GOAL_PROGRESS,
        AppraisalReasonCode.THREAT,
    )

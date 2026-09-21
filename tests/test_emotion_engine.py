from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import math

import pytest

from kagya.body import (
    ArousalContributions,
    EmotionEngineAllostasis,
    EmotionState,
    EmotionTemporalState,
    EmotionUpdateReasonCode,
    ValenceContributions,
)
from kagya.cognition import AppraisalResult


UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def appraisal(
    novelty: float | None = 0.0,
    *,
    valid: bool = True,
    goal_progress: float | None = None,
    threat: float | None = None,
    controllability: float | None = None,
    certainty: float | None = None,
    social_relevance: float | None = None,
    effort_cost: float | None = None,
) -> AppraisalResult:
    return AppraisalResult(
        novelty=novelty if valid else None,
        novelty_valid=valid,
        goal_progress=goal_progress,
        threat=threat,
        controllability=controllability,
        certainty=certainty,
        social_relevance=social_relevance,
        effort_cost=effort_cost,
    )


def test_arousal_is_clamped_to_unit_interval() -> None:
    engine = EmotionEngineAllostasis(EmotionState(arousal=1.0))

    state = engine.update(100.0)

    assert state.arousal == 1.0
    assert 0.0 <= state.arousal <= 1.0


def test_valence_is_clamped_to_signed_unit_interval() -> None:
    engine = EmotionEngineAllostasis(EmotionState(valence=1.0, optimal_loss=0.0))

    state = engine.update(100.0)

    assert state.valence == -1.0
    assert -1.0 <= state.valence <= 1.0


def test_optimal_loss_updates_with_adaptation_rate() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(optimal_loss=1.0),
        adaptation_rate=0.25,
    )

    state = engine.update(3.0)

    assert state.optimal_loss == 1.5


def test_extreme_loss_values_do_not_produce_nan() -> None:
    engine = EmotionEngineAllostasis()


    for loss in [1e308, math.inf, math.nan, -math.inf]:
        state = engine.update(loss)
        assert not math.isnan(state.valence)
        assert not math.isnan(state.arousal)
        assert not math.isnan(state.optimal_loss)


def test_emotion_update_uses_specified_formula() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.5, arousal=0.25, optimal_loss=1.0),
        adaptation_rate=0.1,
    )

    state = engine.update(0.5)

    assert state.arousal == 0.25 * 0.8 + 0.5 * 0.2
    assert state.valence == 0.5 * 0.4 + (1.0 - 0.3 * (0.5 - 1.0) ** 2) * 0.6
    assert state.optimal_loss == 0.9 * 1.0 + 0.1 * 0.5


def test_identical_appraisal_and_configuration_are_deterministic() -> None:
    initial = EmotionState(valence=-0.2, arousal=0.3, optimal_loss=1.0)
    evidence = appraisal(
        0.5,
        goal_progress=0.4,
        threat=0.2,
        controllability=0.8,
        certainty=0.25,
        social_relevance=0.6,
        effort_cost=0.3,
    )
    first = EmotionEngineAllostasis(initial).update_from_appraisal(evidence)
    second = EmotionEngineAllostasis(initial).update_from_appraisal(evidence)

    assert first == second


def test_structured_contributions_use_the_fixed_policy() -> None:
    result = EmotionEngineAllostasis(
        appraisal_response_rate=1.0,
    ).update_from_appraisal(
        appraisal(
            0.5,
            goal_progress=0.5,
            threat=0.25,
            effort_cost=0.4,
            controllability=0.75,
            certainty=0.25,
            social_relevance=0.6,
        )
    )

    assert result.valence_contributions == ValenceContributions(
        goal_progress=0.35,
        threat=-0.2,
        effort_cost=-0.12,
        controllability=0.05,
    )
    assert result.arousal_contributions.novelty == pytest.approx(0.3)
    assert result.arousal_contributions.threat == pytest.approx(0.175)
    assert result.arousal_contributions.effort_cost == pytest.approx(0.12)
    assert result.arousal_contributions.social_relevance == pytest.approx(0.12)
    assert result.arousal_contributions.uncertainty == pytest.approx(0.15)
    assert result.arousal_contributions.low_controllability == pytest.approx(0.05)
    assert result.state.valence == pytest.approx(0.08)
    assert result.state.arousal == pytest.approx(0.915)
    assert result.reasons == (EmotionUpdateReasonCode.APPRAISAL_APPLIED,)


@pytest.mark.parametrize(
    "contribution",
    [
        lambda: ValenceContributions(goal_progress=0.7001),
        lambda: ValenceContributions(threat=-0.8001),
        lambda: ValenceContributions(effort_cost=-0.3001),
        lambda: ValenceContributions(controllability=0.1001),
        lambda: ArousalContributions(novelty=0.6001),
        lambda: ArousalContributions(threat=0.7001),
        lambda: ArousalContributions(effort_cost=0.3001),
        lambda: ArousalContributions(social_relevance=0.2001),
        lambda: ArousalContributions(uncertainty=0.2001),
        lambda: ArousalContributions(low_controllability=0.2001),
    ],
)
def test_contribution_values_are_strictly_policy_bounded(contribution) -> None:
    with pytest.raises(ValueError):
        contribution()


def test_goal_progress_has_more_positive_valence_than_threat() -> None:
    engine = EmotionEngineAllostasis(appraisal_response_rate=1.0)
    goal = engine.update_from_appraisal(appraisal(0.5, goal_progress=1.0))
    threat = EmotionEngineAllostasis(appraisal_response_rate=1.0).update_from_appraisal(
        appraisal(0.5, threat=1.0)
    )

    assert goal.state.valence > threat.state.valence
    assert threat.state.arousal > goal.state.arousal


def test_invalid_novelty_alone_is_an_emotion_no_op() -> None:
    engine = EmotionEngineAllostasis(EmotionState(valence=0.4, arousal=0.6))
    before = engine.state

    result = engine.update_from_appraisal(appraisal(valid=False))

    assert result.state == before
    assert result.valence_contributions == ValenceContributions()
    assert result.arousal_contributions == ArousalContributions()
    assert result.reasons == (
        EmotionUpdateReasonCode.NOVELTY_OMITTED,
        EmotionUpdateReasonCode.NO_MATERIAL_APPRAISAL,
    )


def test_invalid_novelty_does_not_suppress_valid_threat() -> None:
    result = EmotionEngineAllostasis(
        appraisal_response_rate=1.0,
    ).update_from_appraisal(appraisal(valid=False, threat=1.0))

    assert result.valence_contributions.threat == -0.8
    assert result.arousal_contributions.threat == 0.7
    assert result.state.valence == -0.8
    assert result.state.arousal == 0.7
    assert result.reasons == (
        EmotionUpdateReasonCode.NOVELTY_OMITTED,
        EmotionUpdateReasonCode.APPRAISAL_APPLIED,
    )


def test_missing_certainty_and_controllability_contribute_zero() -> None:
    missing = EmotionEngineAllostasis(appraisal_response_rate=1.0).update_from_appraisal(
        appraisal(0.0)
    )
    explicit = EmotionEngineAllostasis(appraisal_response_rate=1.0).update_from_appraisal(
        appraisal(0.0, certainty=0.25, controllability=0.0)
    )

    assert missing.arousal_contributions.uncertainty == 0.0
    assert missing.arousal_contributions.low_controllability == 0.0
    assert missing.valence_contributions.controllability == 0.0
    assert explicit.arousal_contributions.uncertainty == pytest.approx(0.15)
    assert explicit.arousal_contributions.low_controllability == pytest.approx(0.2)
    assert explicit.valence_contributions.controllability == pytest.approx(-0.1)


def test_zero_only_evidence_does_not_drift_toward_rest() -> None:
    engine = EmotionEngineAllostasis(EmotionState(valence=0.6, arousal=0.8))
    before = engine.state

    result = engine.update_from_appraisal(
        appraisal(
            0.0,
            goal_progress=0.0,
            threat=0.0,
            effort_cost=0.0,
            certainty=1.0,
            social_relevance=0.0,
        )
    )

    assert result.state == before
    assert result.reasons == (EmotionUpdateReasonCode.NO_MATERIAL_APPRAISAL,)


def test_response_rate_zero_is_an_appraisal_no_op() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.2, arousal=0.3),
        appraisal_response_rate=0.0,
    )
    result = engine.update_from_appraisal(appraisal(1.0, threat=1.0))

    assert result.state.valence == 0.2
    assert result.state.arousal == 0.3
    assert result.reasons == (EmotionUpdateReasonCode.APPRAISAL_APPLIED,)


def test_response_rate_one_reaches_bounded_contribution_targets() -> None:
    result = EmotionEngineAllostasis(
        appraisal_response_rate=1.0,
    ).update_from_appraisal(
        appraisal(
            1.0,
            threat=1.0,
            effort_cost=1.0,
            controllability=0.0,
            certainty=0.0,
            social_relevance=1.0,
            goal_progress=-1.0,
        )
    )

    assert result.state.valence == -1.0
    assert result.state.arousal == 1.0
    assert all(
        math.isfinite(value)
        for value in (
            result.valence_contributions.goal_progress,
            result.valence_contributions.threat,
            result.valence_contributions.effort_cost,
            result.valence_contributions.controllability,
            result.arousal_contributions.novelty,
            result.arousal_contributions.threat,
            result.arousal_contributions.effort_cost,
            result.arousal_contributions.social_relevance,
            result.arousal_contributions.uncertainty,
            result.arousal_contributions.low_controllability,
        )
    )


def test_response_rate_one_reaches_non_clamped_targets_exactly() -> None:
    result = EmotionEngineAllostasis(
        appraisal_response_rate=1.0,
    ).update_from_appraisal(
        appraisal(
            0.5,
            goal_progress=0.5,
            threat=0.25,
            effort_cost=0.4,
            controllability=0.75,
            certainty=0.25,
            social_relevance=0.6,
        )
    )

    assert result.state.valence == math.fsum(
        (
            result.valence_contributions.goal_progress,
            result.valence_contributions.threat,
            result.valence_contributions.effort_cost,
            result.valence_contributions.controllability,
        )
    )
    assert result.state.arousal == math.fsum(
        (
            result.arousal_contributions.novelty,
            result.arousal_contributions.threat,
            result.arousal_contributions.effort_cost,
            result.arousal_contributions.social_relevance,
            result.arousal_contributions.uncertainty,
            result.arousal_contributions.low_controllability,
        )
    )


def test_primary_loss_is_compatibility_only() -> None:
    evidence = appraisal(0.5, goal_progress=0.5, threat=0.2)
    unchanged = EmotionEngineAllostasis(
        EmotionState(optimal_loss=1.0), adaptation_rate=0.1
    ).update_from_appraisal(evidence)
    updated = EmotionEngineAllostasis(
        EmotionState(optimal_loss=1.0), adaptation_rate=0.1
    ).update_from_appraisal(evidence, primary_loss=3.0)

    assert unchanged.state.optimal_loss == 1.0
    assert updated.state.optimal_loss == pytest.approx(1.2)
    assert updated.valence_contributions == unchanged.valence_contributions
    assert updated.arousal_contributions == unchanged.arousal_contributions
    assert updated.state.valence == unchanged.state.valence
    assert updated.state.arousal == unchanged.state.arousal


@pytest.mark.parametrize("invalid_loss", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_primary_loss_fails_atomically(invalid_loss: float) -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.2, arousal=0.3, optimal_loss=1.0),
        temporal_state=EmotionTemporalState(T0),
    )
    before_state = engine.state
    before_temporal = engine.temporal_state

    with pytest.raises((TypeError, ValueError)):
        engine.update_from_appraisal(appraisal(0.5, threat=1.0), primary_loss=invalid_loss)

    assert engine.state == before_state
    assert engine.temporal_state == before_temporal


def test_advance_time_zero_is_an_exact_no_op() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.4, arousal=0.6, optimal_loss=2.0),
        temporal_state=EmotionTemporalState(T0),
    )
    before = engine.state

    result = engine.advance_time(0.0)

    assert result.state == before
    assert engine.state == before
    assert engine.temporal_state == EmotionTemporalState(T0)
    assert result.reasons == (EmotionUpdateReasonCode.TIME_RECOVERY,)


def test_advance_time_moves_monotonically_toward_resting_state() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.8, arousal=0.9, optimal_loss=2.0),
        resting_valence=-0.2,
        resting_arousal=0.1,
        valence_recovery_rate=0.1,
        arousal_recovery_rate=0.2,
    )

    result = engine.advance_time(1.0)

    assert result.state.valence == pytest.approx(-0.2 + 1.0 * math.exp(-0.1))
    assert result.state.arousal == pytest.approx(0.1 + 0.8 * math.exp(-0.2))
    assert -0.2 < result.state.valence < 0.8
    assert 0.1 < result.state.arousal < 0.9
    assert result.state.optimal_loss == 2.0


def test_long_recovery_converges_without_overshoot() -> None:
    result = EmotionEngineAllostasis(
        EmotionState(valence=1.0, arousal=1.0),
        resting_valence=-1.0,
        resting_arousal=0.0,
        valence_recovery_rate=0.01,
        arousal_recovery_rate=0.02,
    ).advance_time(1e308)

    assert result.state == EmotionState(valence=-1.0, arousal=0.0)
    assert math.isfinite(result.state.valence)
    assert math.isfinite(result.state.arousal)


def test_zero_recovery_rate_leaves_that_axis_unchanged() -> None:
    result = EmotionEngineAllostasis(
        EmotionState(valence=0.8, arousal=0.8),
        resting_valence=-0.8,
        resting_arousal=0.0,
        valence_recovery_rate=0.0,
        arousal_recovery_rate=0.1,
    ).advance_time(10.0)

    assert result.state.valence == 0.8
    assert result.state.arousal < 0.8


@pytest.mark.parametrize("elapsed", [-1.0, math.nan, math.inf])
def test_invalid_elapsed_fails_atomically(elapsed: float) -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.4, arousal=0.6),
        temporal_state=EmotionTemporalState(T0),
    )
    before_state = engine.state
    before_temporal = engine.temporal_state

    with pytest.raises((TypeError, ValueError)):
        engine.advance_time(elapsed)

    assert engine.state == before_state
    assert engine.temporal_state == before_temporal


def test_first_advance_to_initializes_timeline_without_recovery() -> None:
    engine = EmotionEngineAllostasis(EmotionState(valence=0.4, arousal=0.6))

    result = engine.advance_to(T0)

    assert result.state == EmotionState(valence=0.4, arousal=0.6)
    assert result.reasons == (EmotionUpdateReasonCode.TIMELINE_INITIALIZED,)
    assert engine.temporal_state == EmotionTemporalState(T0)


def test_advance_to_applies_exact_elapsed_interval() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.8, arousal=0.6),
        resting_valence=0.0,
        resting_arousal=0.0,
        valence_recovery_rate=0.1,
        arousal_recovery_rate=0.2,
        temporal_state=EmotionTemporalState(T0),
    )

    result = engine.advance_to(T0 + timedelta(seconds=2.0))

    assert result.state.valence == pytest.approx(0.8 * math.exp(-0.2))
    assert result.state.arousal == pytest.approx(0.6 * math.exp(-0.4))
    assert engine.temporal_state == EmotionTemporalState(T0 + timedelta(seconds=2.0))
    assert result.reasons == (EmotionUpdateReasonCode.TIME_RECOVERY,)


def test_equal_timestamp_is_a_zero_time_no_op() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.4, arousal=0.6),
        temporal_state=EmotionTemporalState(T0),
    )
    before = engine.state

    result = engine.advance_to(T0)

    assert result.state == before
    assert engine.temporal_state == EmotionTemporalState(T0)


def test_regressed_timestamp_fails_atomically() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.4, arousal=0.6),
        temporal_state=EmotionTemporalState(T0),
    )
    before_state = engine.state
    before_temporal = engine.temporal_state

    with pytest.raises(ValueError, match="regress"):
        engine.advance_to(T0 - timedelta(seconds=1.0))

    assert engine.state == before_state
    assert engine.temporal_state == before_temporal


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_non_utc_timestamp_fails_atomically(invalid_timestamp: datetime) -> None:
    engine = EmotionEngineAllostasis(temporal_state=EmotionTemporalState(T0))
    before_state = engine.state
    before_temporal = engine.temporal_state

    with pytest.raises((TypeError, ValueError)):
        engine.advance_to(invalid_timestamp)

    assert engine.state == before_state
    assert engine.temporal_state == before_temporal


def test_advance_to_uses_injected_clock_when_now_is_omitted() -> None:
    current = [T0]
    engine = EmotionEngineAllostasis(clock=lambda: current[0])

    first = engine.advance_to()
    current[0] = T0 + timedelta(seconds=5.0)
    second = engine.advance_to()

    assert first.reasons == (EmotionUpdateReasonCode.TIMELINE_INITIALIZED,)
    assert second.reasons == (EmotionUpdateReasonCode.TIME_RECOVERY,)
    assert engine.temporal_state == EmotionTemporalState(current[0])


def test_temporal_and_update_values_are_immutable_and_reads_are_pure() -> None:
    temporal = EmotionTemporalState(T0)
    engine = EmotionEngineAllostasis(temporal_state=temporal)
    before = engine.state

    with pytest.raises(FrozenInstanceError):
        temporal.last_update_at = T0 + timedelta(seconds=1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        engine.advance_time(0.0).state.valence = 0.5  # type: ignore[misc]

    assert repr(engine.state)
    assert repr(engine.temporal_state)
    assert engine.state == before

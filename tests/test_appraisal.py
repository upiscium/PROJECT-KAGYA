import math

import pytest

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.cognition import (
    AppraisalSignals,
    CognitiveAppraiser,
    LossMeasurement,
    SurprisalCalculator,
)
from kagya.models import DummyProvider


class InvalidLossProvider(DummyProvider):
    def calculate_loss(self, context_text: str, target_text: str) -> float:
        return math.nan


class FailingLossProvider(DummyProvider):
    def calculate_loss(self, context_text: str, target_text: str) -> float:
        raise RuntimeError("loss unavailable")


def test_calibrated_novelty_is_bounded_and_model_specific() -> None:
    calculator = SurprisalCalculator(DummyProvider())

    first = calculator.measure("", "hello", model_key="model-a")
    second = calculator.measure("", "hello", model_key="model-b")

    assert first.valid is True
    assert first.calibrated_novelty is not None
    assert 0.0 <= first.calibrated_novelty <= 1.0
    assert second.calibrated_novelty == first.calibrated_novelty
    assert set(calculator.history) == {"model-a", "model-b"}


@pytest.mark.parametrize(
    ("provider", "reason"),
    [(InvalidLossProvider(), "non_finite_loss"), (FailingLossProvider(), "provider_error")],
)
def test_invalid_loss_is_not_zero_or_added_to_history(provider, reason: str) -> None:
    calculator = SurprisalCalculator(provider)

    measurement = calculator.measure("context", "target", model_key="model")

    assert measurement.valid is False
    assert measurement.raw_loss is None
    assert measurement.calibrated_novelty is None
    assert measurement.invalid_reason == reason
    assert calculator.history == {}


def test_same_novelty_with_threat_and_goal_progress_changes_emotion_differently() -> None:
    measurement = LossMeasurement(0.5, 0.5, 1, "model", True, None, 0.5)
    appraiser = CognitiveAppraiser()
    positive = appraiser.assess(
        measurement, AppraisalSignals(goal_progress=1.0, threat=0.0)
    )
    threatened = appraiser.assess(
        measurement, AppraisalSignals(goal_progress=0.0, threat=1.0)
    )
    positive_engine = EmotionEngineAllostasis(EmotionState())
    threat_engine = EmotionEngineAllostasis(EmotionState())

    positive_state = positive_engine.update_from_appraisal(positive).state
    threatened_state = threat_engine.update_from_appraisal(threatened).state

    assert positive_state.valence > threatened_state.valence
    assert threatened_state.arousal > positive_state.arousal


def test_invalid_novelty_is_omitted_from_emotion_update() -> None:
    measurement = LossMeasurement(None, None, None, "model", False, "provider_error", None)
    appraisal = CognitiveAppraiser().assess(measurement)

    update = EmotionEngineAllostasis(EmotionState()).update_from_appraisal(appraisal)

    assert update.arousal_contributions["novelty"] == 0.0
    assert "novelty_omitted" in update.reasons
    assert update.state.valence <= 0.1


def test_elapsed_time_recovers_without_overshoot() -> None:
    engine = EmotionEngineAllostasis(
        EmotionState(valence=-0.8, arousal=0.9),
        resting_valence=0.0,
        resting_arousal=0.0,
        valence_recovery_rate=0.1,
        arousal_recovery_rate=0.2,
    )

    state = engine.advance_time(10.0).state

    assert -0.8 < state.valence <= 0.0
    assert 0.0 <= state.arousal < 0.9
    with pytest.raises(ValueError):
        engine.advance_time(-1.0)

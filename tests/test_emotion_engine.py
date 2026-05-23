import math

from kagya.body import EmotionEngineAllostasis, EmotionState


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

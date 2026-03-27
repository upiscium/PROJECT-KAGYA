from project_kagya import EmotionEngineAllostasis


def test_update_uses_specified_formula() -> None:
    engine = EmotionEngineAllostasis(
        optimal_loss=2.5,
        adaptation_rate=0.15,
        valence=0.2,
        arousal=0.1,
    )

    state = engine.update(3.0)

    assert state.arousal == 0.6800000000000002
    assert state.valence == 0.635
    assert state.optimal_loss == 2.575


def test_update_clamps_state_values() -> None:
    engine = EmotionEngineAllostasis(
        optimal_loss=0.0,
        adaptation_rate=0.15,
        valence=1.0,
        arousal=1.0,
    )

    state = engine.update(10.0)

    assert state.arousal == 1.0
    assert state.valence == -1.0

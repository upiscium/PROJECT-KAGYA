from project_kagya.emotion_engine import EmotionEngineAllostasis


def test_emotion_engine_updates_state() -> None:
    engine = EmotionEngineAllostasis()
    state = engine.update(loss=3.0, valence=0.2, arousal=0.5)

    assert round(state.arousal, 3) == 1.0
    assert round(state.valence, 3) == 0.635
    assert round(state.optimal_loss, 3) == 2.575


def test_emotion_engine_clamps_values() -> None:
    engine = EmotionEngineAllostasis()
    state = engine.update(loss=100.0, valence=10.0, arousal=10.0)

    assert state.valence <= 1.0
    assert state.arousal <= 1.0

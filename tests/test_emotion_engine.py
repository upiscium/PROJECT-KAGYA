from __future__ import annotations

import math

import pytest

from project_kagya.emotion_engine import EmotionEngineAllostasis


def test_emotion_engine_starts_with_specified_defaults() -> None:
    engine = EmotionEngineAllostasis()

    state = engine.get_state()

    assert state.valence == 0.0
    assert state.arousal == 0.0
    assert state.optimal_loss == 2.5


def test_emotion_engine_updates_state_by_specified_formula() -> None:
    engine = EmotionEngineAllostasis()

    state = engine.update(1.5)

    assert math.isclose(state.arousal, 0.3)
    assert math.isclose(state.valence, 0.42, rel_tol=1e-9)
    assert math.isclose(state.optimal_loss, 2.35)


def test_emotion_engine_clamps_valence_and_arousal() -> None:
    engine = EmotionEngineAllostasis(valence=2.0, arousal=2.0, optimal_loss=2.5)

    state = engine.update(100.0)

    assert state.valence == -1.0
    assert state.arousal == 1.0


def test_emotion_engine_rejects_non_numeric_loss() -> None:
    engine = EmotionEngineAllostasis()

    with pytest.raises(TypeError, match="loss must be numeric"):
        engine.update("oops")  # type: ignore[arg-type]

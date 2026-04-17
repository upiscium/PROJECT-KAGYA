from __future__ import annotations

import math

import pytest

from project_kagya.embodied_emotion import EmbodiedEmotion, EmbodiedEmotionState
from project_kagya.emotion_engine import EmotionState


def test_update_body_state_changes_with_conversation_event() -> None:
    embodied = EmbodiedEmotion()

    state = embodied.update_body_state({"type": "conversation", "intensity": 2.0})

    assert state.fatigue > 0.0
    assert state.load > 0.0
    assert state.recovery < 0.0 or math.isclose(state.recovery, 0.0)


def test_update_body_state_changes_with_sleep_event() -> None:
    embodied = EmbodiedEmotion()
    embodied.update_body_state({"type": "conversation", "intensity": 2.0})

    state = embodied.update_body_state({"type": "sleep", "duration": 2.0})

    assert state.recovery > 0.0
    assert state.stability > 0.0


def test_modulate_emotion_reflects_body_state() -> None:
    embodied = EmbodiedEmotion()
    embodied.update_body_state({"type": "conversation", "intensity": 2.0})

    state = embodied.modulate_emotion(1.5, EmotionState(0.0, 0.0, 2.5))

    assert -1.0 <= state.valence <= 1.0
    assert 0.0 <= state.arousal <= 1.0


def test_get_body_state_returns_current_state() -> None:
    embodied = EmbodiedEmotion()

    state = embodied.get_body_state()

    assert isinstance(state, EmbodiedEmotionState)


def test_update_body_state_rejects_unknown_event_type() -> None:
    embodied = EmbodiedEmotion()

    with pytest.raises(ValueError, match="unsupported event type"):
        embodied.update_body_state({"type": "unknown"})

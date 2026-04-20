from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from project_kagya.emotion_engine import (  # type: ignore[import-untyped]
    EmotionEngineAllostasis,
    EmotionState,
)


@dataclass(slots=True)
class EmbodiedEmotionState:
    fatigue: float = 0.0
    load: float = 0.0
    stability: float = 0.0
    recovery: float = 0.0


class EmbodiedEmotion:
    def __init__(self) -> None:
        self._body_state = EmbodiedEmotionState()
        self._emotion_engine = EmotionEngineAllostasis()

    def update_body_state(self, event: dict[str, object]) -> EmbodiedEmotionState:
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")
        if not event:
            return self._body_state

        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise ValueError("event must include a string type")

        if event_type == "conversation":
            intensity = self._numeric(event.get("intensity", 1.0), default=1.0)
            self._body_state.fatigue = self._clamp(
                self._body_state.fatigue + 0.05 * intensity
            )
            self._body_state.load = self._clamp(
                self._body_state.load + 0.04 * intensity
            )
            self._body_state.stability = self._clamp(
                self._body_state.stability + 0.02 * intensity
            )
            self._body_state.recovery = self._clamp(
                self._body_state.recovery - 0.03 * intensity
            )
        elif event_type == "sleep":
            duration = self._numeric(event.get("duration", 1.0), default=1.0)
            self._body_state.fatigue = self._clamp(
                self._body_state.fatigue - 0.2 * duration
            )
            self._body_state.load = self._clamp(self._body_state.load - 0.1 * duration)
            self._body_state.stability = self._clamp(
                self._body_state.stability + 0.1 * duration
            )
            self._body_state.recovery = self._clamp(
                self._body_state.recovery + 0.25 * duration
            )
        elif event_type == "failure":
            severity = self._numeric(event.get("severity", 1.0), default=1.0)
            self._body_state.fatigue = self._clamp(
                self._body_state.fatigue + 0.1 * severity
            )
            self._body_state.load = self._clamp(self._body_state.load + 0.15 * severity)
            self._body_state.stability = self._clamp(
                self._body_state.stability - 0.08 * severity
            )
        elif event_type == "success":
            reward = self._numeric(event.get("reward", 1.0), default=1.0)
            self._body_state.stability = self._clamp(
                self._body_state.stability + 0.12 * reward
            )
            self._body_state.recovery = self._clamp(
                self._body_state.recovery + 0.05 * reward
            )
            self._body_state.load = self._clamp(self._body_state.load - 0.04 * reward)
        else:
            raise ValueError(f"unsupported event type: {event_type}")

        return self._body_state

    def modulate_emotion(
        self, loss: float, emotion_state: EmotionState
    ) -> EmotionState:
        if not self._is_number(loss):
            raise TypeError("loss must be numeric")
        if not isinstance(emotion_state, EmotionState):
            raise TypeError("emotion_state must be EmotionState")

        body = self._body_state
        body_load = 1.0 + (body.load * 0.3)
        body_fatigue = 1.0 - (body.fatigue * 0.25)
        body_recovery = 1.0 - (body.recovery * 0.15)
        effective_loss = float(loss) * body_load * body_fatigue * body_recovery

        valence_scale = 1.0 - (body.stability * 0.5)
        arousal_bias = (body.load * 0.2) - (body.recovery * 0.1) - (body.fatigue * 0.1)

        base = self._emotion_engine
        current_valence = emotion_state.valence * valence_scale
        current_arousal = emotion_state.arousal + arousal_bias
        current_optimal_loss = emotion_state.optimal_loss

        updated = EmotionState(
            valence=self._clamp(
                current_valence * 0.4
                + (1.0 - 0.3 * (effective_loss - current_optimal_loss) ** 2) * 0.6,
                -1.0,
                1.0,
            ),
            arousal=self._clamp(current_arousal * 0.8 + effective_loss * 0.2, 0.0, 1.0),
            optimal_loss=(1.0 - base.adaptation_rate) * current_optimal_loss
            + base.adaptation_rate * effective_loss,
        )
        return updated

    def get_body_state(self) -> EmbodiedEmotionState:
        return self._body_state

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _numeric(value: object, default: float) -> float:
        if value is None:
            return default
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError("event values must be numeric")
        return float(value)

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

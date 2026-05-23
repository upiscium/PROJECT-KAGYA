"""Allostatic emotion-state update logic."""

from dataclasses import dataclass
import math


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    optimal_loss: float = 1.0


class EmotionEngineAllostasis:
    """Update emotion state from prediction loss."""

    def __init__(
        self,
        state: EmotionState | None = None,
        adaptation_rate: float = 0.05,
    ) -> None:
        self.state = state or EmotionState()
        self.adaptation_rate = _clamp(adaptation_rate, 0.0, 1.0)

    def update(self, loss: float) -> EmotionState:
        stable_loss = _finite_or_default(loss, default=0.0)
        arousal = _clamp(self.state.arousal * 0.8 + stable_loss * 0.2, 0.0, 1.0)
        diff = stable_loss - self.state.optimal_loss
        squared_diff = _safe_square(diff)
        wellbeing = 1.0 - 0.3 * squared_diff
        valence = _clamp(self.state.valence * 0.4 + wellbeing * 0.6, -1.0, 1.0)
        optimal_loss = (
            (1.0 - self.adaptation_rate) * self.state.optimal_loss
            + self.adaptation_rate * stable_loss
        )
        optimal_loss = _finite_or_default(optimal_loss, default=self.state.optimal_loss)
        self.state = EmotionState(
            valence=valence,
            arousal=arousal,
            optimal_loss=optimal_loss,
        )
        return self.state


def _safe_square(value: float) -> float:
    try:
        squared = value * value
    except OverflowError:
        return math.inf
    return _finite_or_default(squared, default=math.inf)


def _finite_or_default(value: float, default: float) -> float:
    return value if math.isfinite(value) else default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    stable_value = _finite_or_default(value, default=minimum)
    return max(minimum, min(maximum, stable_value))

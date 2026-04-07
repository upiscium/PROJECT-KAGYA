from __future__ import annotations

from dataclasses import dataclass
from numbers import Real


@dataclass(slots=True)
class EmotionState:
    valence: float
    arousal: float
    optimal_loss: float


class EmotionEngineAllostasis:
    def __init__(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        optimal_loss: float = 2.5,
        adaptation_rate: float = 0.15,
    ) -> None:
        self._state = EmotionState(
            valence=valence,
            arousal=arousal,
            optimal_loss=optimal_loss,
        )
        self.adaptation_rate = adaptation_rate

    def update(self, loss: float) -> EmotionState:
        if not self._is_number(loss):
            raise TypeError("loss must be numeric")

        current = self._state
        arousal = self._clamp(current.arousal * 0.8 + float(loss) * 0.2, 0.0, 1.0)
        wundt = 1.0 - 0.3 * (float(loss) - current.optimal_loss) ** 2
        valence = self._clamp(current.valence * 0.4 + wundt * 0.6, -1.0, 1.0)
        optimal_loss = (
            1.0 - self.adaptation_rate
        ) * current.optimal_loss + self.adaptation_rate * float(loss)

        self._state = EmotionState(
            valence=valence,
            arousal=arousal,
            optimal_loss=optimal_loss,
        )
        return self._state

    def get_state(self) -> EmotionState:
        return self._state

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

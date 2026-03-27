from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    optimal_loss: float = 2.5


class EmotionEngineAllostasis:
    def __init__(
        self,
        optimal_loss: float = 2.5,
        adaptation_rate: float = 0.15,
        valence: float = 0.0,
        arousal: float = 0.0,
    ) -> None:
        self.state = EmotionState(
            valence=valence,
            arousal=arousal,
            optimal_loss=optimal_loss,
        )
        self.adaptation_rate = adaptation_rate

    def update(self, loss: float) -> EmotionState:
        current = self.state
        arousal_new = _clamp(current.arousal * 0.8 + loss * 0.2, 0.0, 1.0)
        wound = 1.0 - 0.3 * (loss - current.optimal_loss) ** 2
        valence_new = _clamp(current.valence * 0.4 + wound * 0.6, -1.0, 1.0)
        optimal_loss_new = (
            1.0 - self.adaptation_rate
        ) * current.optimal_loss + self.adaptation_rate * loss

        self.state = EmotionState(
            valence=valence_new,
            arousal=arousal_new,
            optimal_loss=optimal_loss_new,
        )
        return self.state

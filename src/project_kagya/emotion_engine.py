from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    optimal_loss: float = 2.5


class EmotionEngineAllostasis:
    def __init__(
        self, optimal_loss: float = 2.5, adaptation_rate: float = 0.15
    ) -> None:
        self.optimal_loss = optimal_loss
        self.adaptation_rate = adaptation_rate

    def update(self, loss: float, valence: float, arousal: float) -> EmotionState:
        arousal_new = _clamp(arousal * 0.8 + loss * 0.2, 0.0, 1.0)
        wundt = 1.0 - 0.3 * (loss - self.optimal_loss) ** 2
        valence_new = _clamp(valence * 0.4 + wundt * 0.6, -1.0, 1.0)
        self.optimal_loss = (
            1.0 - self.adaptation_rate
        ) * self.optimal_loss + self.adaptation_rate * loss
        return EmotionState(
            valence=valence_new, arousal=arousal_new, optimal_loss=self.optimal_loss
        )

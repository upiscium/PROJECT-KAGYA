"""Allostatic emotion-state update logic."""

from dataclasses import dataclass
import math

from kagya.cognition import AppraisalResult


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    optimal_loss: float = 1.0


@dataclass(frozen=True)
class EmotionUpdate:
    state: EmotionState
    valence_contributions: dict[str, float]
    arousal_contributions: dict[str, float]
    reasons: tuple[str, ...]


class EmotionEngineAllostasis:
    """Update emotion state from prediction loss."""

    def __init__(
        self,
        state: EmotionState | None = None,
        adaptation_rate: float = 0.05,
        response_rate: float = 0.4,
        resting_valence: float = 0.0,
        resting_arousal: float = 0.0,
        valence_recovery_rate: float = 0.01,
        arousal_recovery_rate: float = 0.02,
    ) -> None:
        self.state = state or EmotionState()
        self.adaptation_rate = _clamp(adaptation_rate, 0.0, 1.0)
        self.response_rate = _clamp(response_rate, 0.0, 1.0)
        self.resting_valence = _clamp(resting_valence, -1.0, 1.0)
        self.resting_arousal = _clamp(resting_arousal, 0.0, 1.0)
        self.valence_recovery_rate = max(0.0, valence_recovery_rate)
        self.arousal_recovery_rate = max(0.0, arousal_recovery_rate)

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

    def update_from_appraisal(self, appraisal: AppraisalResult) -> EmotionUpdate:
        valence_contributions = {
            "goal_progress": 0.7 * appraisal.goal_progress,
            "threat": -0.8 * appraisal.threat,
            "effort_cost": -0.3 * appraisal.effort_cost,
            "controllability": 0.2 * (appraisal.controllability - 0.5),
        }
        arousal_contributions = {
            "novelty": 0.0 if appraisal.novelty is None else 0.6 * appraisal.novelty,
            "threat": 0.7 * appraisal.threat,
            "effort_cost": 0.3 * appraisal.effort_cost,
            "social_relevance": 0.2 * appraisal.social_relevance,
            "uncertainty": 0.2 * (1.0 - appraisal.certainty),
            "low_controllability": 0.2 * (1.0 - appraisal.controllability),
        }
        target_valence = _clamp(sum(valence_contributions.values()), -1.0, 1.0)
        target_arousal = _clamp(sum(arousal_contributions.values()), 0.0, 1.0)
        valence = _approach(self.state.valence, target_valence, self.response_rate)
        arousal = _approach(self.state.arousal, target_arousal, self.response_rate)
        self.state = EmotionState(
            valence=_clamp(valence, -1.0, 1.0),
            arousal=_clamp(arousal, 0.0, 1.0),
            optimal_loss=self.state.optimal_loss,
        )
        reasons = list(appraisal.reasons)
        if not appraisal.novelty_valid:
            reasons.append("novelty_omitted")
        return EmotionUpdate(
            state=self.state,
            valence_contributions=valence_contributions,
            arousal_contributions=arousal_contributions,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def advance_time(self, elapsed_seconds: float) -> EmotionUpdate:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        valence_rate = 1.0 - math.exp(-self.valence_recovery_rate * elapsed_seconds)
        arousal_rate = 1.0 - math.exp(-self.arousal_recovery_rate * elapsed_seconds)
        self.state = EmotionState(
            valence=_approach(self.state.valence, self.resting_valence, valence_rate),
            arousal=_approach(self.state.arousal, self.resting_arousal, arousal_rate),
            optimal_loss=self.state.optimal_loss,
        )
        return EmotionUpdate(
            state=self.state,
            valence_contributions={"recovery": self.state.valence},
            arousal_contributions={"recovery": self.state.arousal},
            reasons=("time_recovery",),
        )


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


def _approach(current: float, target: float, rate: float) -> float:
    return current + (target - current) * _clamp(rate, 0.0, 1.0)

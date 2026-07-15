"""Structured cognitive appraisal and loss calibration."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LossMeasurement:
    raw_loss: float | None
    mean_token_loss: float | None
    target_token_count: int | None
    model_key: str
    valid: bool
    invalid_reason: str | None
    calibrated_novelty: float | None


@dataclass(frozen=True)
class AppraisalSignals:
    goal_progress: float = 0.0
    threat: float = 0.0
    controllability: float = 0.5
    certainty: float = 0.5
    social_relevance: float = 0.0
    effort_cost: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.goal_progress <= 1.0:
            raise ValueError("goal_progress must be between -1 and 1")
        for name in (
            "threat",
            "controllability",
            "certainty",
            "social_relevance",
            "effort_cost",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")


@dataclass(frozen=True)
class AppraisalResult:
    novelty: float | None
    goal_progress: float
    threat: float
    controllability: float
    certainty: float
    social_relevance: float
    effort_cost: float
    novelty_valid: bool
    reasons: tuple[str, ...]


class CognitiveAppraiser:
    def assess(
        self,
        measurement: LossMeasurement,
        signals: AppraisalSignals | None = None,
    ) -> AppraisalResult:
        current = signals or AppraisalSignals()
        reasons = [
            "novelty_measured" if measurement.valid else "novelty_invalid",
        ]
        if measurement.invalid_reason:
            reasons.append(measurement.invalid_reason)
        if current.goal_progress > 0:
            reasons.append("goal_progress")
        elif current.goal_progress < 0:
            reasons.append("goal_setback")
        if current.threat > 0:
            reasons.append("threat")
        return AppraisalResult(
            novelty=measurement.calibrated_novelty,
            goal_progress=current.goal_progress,
            threat=current.threat,
            controllability=current.controllability,
            certainty=current.certainty,
            social_relevance=current.social_relevance,
            effort_cost=current.effort_cost,
            novelty_valid=measurement.valid,
            reasons=tuple(reasons),
        )


def bounded_sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-value))

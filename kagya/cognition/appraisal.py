"""Pure, structured appraisal of already measured typed evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from kagya.cognition.surprisal_calculator import LossMeasurement


class AppraisalReasonCode(str, Enum):
    """The finite vocabulary used to explain an appraisal."""

    NOVELTY_MEASURED = "novelty_measured"
    NOVELTY_INVALID = "novelty_invalid"
    GOAL_PROGRESS = "goal_progress"
    GOAL_SETBACK = "goal_setback"
    THREAT = "threat"


AppraisalReason = AppraisalReasonCode

_SIGNAL_NAMES = (
    "goal_progress",
    "threat",
    "controllability",
    "certainty",
    "social_relevance",
    "effort_cost",
)
_MAX_REASONS = len(AppraisalReasonCode)


def _bounded(
    value: float | None,
    name: str,
    *,
    lower: float,
    upper: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


@dataclass(frozen=True, slots=True)
class AppraisalSignals:
    """Explicit appraisal evidence; unavailable dimensions remain ``None``."""

    goal_progress: float | None = None
    threat: float | None = None
    controllability: float | None = None
    certainty: float | None = None
    social_relevance: float | None = None
    effort_cost: float | None = None

    def __post_init__(self) -> None:
        for name in _SIGNAL_NAMES:
            lower, upper = (-1.0, 1.0) if name == "goal_progress" else (0.0, 1.0)
            value = _bounded(
                getattr(self, name), name, lower=lower, upper=upper
            )
            object.__setattr__(self, name, value)


def _validate_reasons(
    reasons: tuple[AppraisalReasonCode, ...],
) -> tuple[AppraisalReasonCode, ...]:
    if not isinstance(reasons, tuple):
        raise TypeError("reasons must be a tuple")
    if len(reasons) > _MAX_REASONS:
        raise ValueError("too many appraisal reasons")
    if any(not isinstance(reason, AppraisalReasonCode) for reason in reasons):
        raise TypeError("reasons must contain only AppraisalReasonCode values")
    if len(set(reasons)) != len(reasons):
        raise ValueError("reasons must not contain duplicates")
    return reasons


@dataclass(frozen=True, slots=True)
class AppraisalResult:
    """Immutable output retaining novelty validity and every supplied signal."""

    novelty: float | None
    novelty_valid: bool
    goal_progress: float | None = None
    threat: float | None = None
    controllability: float | None = None
    certainty: float | None = None
    social_relevance: float | None = None
    effort_cost: float | None = None
    reasons: tuple[AppraisalReasonCode, ...] = ()

    def __post_init__(self) -> None:
        if type(self.novelty_valid) is not bool:
            raise TypeError("novelty_valid must be bool")
        novelty = _bounded(self.novelty, "novelty", lower=0.0, upper=1.0)
        if self.novelty_valid != (novelty is not None):
            raise ValueError("novelty and novelty_valid disagree")
        object.__setattr__(self, "novelty", novelty)
        signals = AppraisalSignals(
            self.goal_progress,
            self.threat,
            self.controllability,
            self.certainty,
            self.social_relevance,
            self.effort_cost,
        )
        for name in _SIGNAL_NAMES:
            object.__setattr__(self, name, getattr(signals, name))
        _validate_reasons(self.reasons)


class CognitiveAppraiser:
    """Turn supplied evidence into a deterministic structured appraisal."""

    __slots__ = ()

    def appraise(
        self, measurement: LossMeasurement, signals: AppraisalSignals
    ) -> AppraisalResult:
        if not isinstance(measurement, LossMeasurement):
            raise TypeError("measurement must be a LossMeasurement")
        if not isinstance(signals, AppraisalSignals):
            raise TypeError("signals must be AppraisalSignals")

        reasons: list[AppraisalReasonCode] = [
            AppraisalReasonCode.NOVELTY_MEASURED
            if measurement.valid
            else AppraisalReasonCode.NOVELTY_INVALID
        ]
        if signals.goal_progress is not None:
            if signals.goal_progress > 0.0:
                reasons.append(AppraisalReasonCode.GOAL_PROGRESS)
            elif signals.goal_progress < 0.0:
                reasons.append(AppraisalReasonCode.GOAL_SETBACK)
        if signals.threat is not None and signals.threat > 0.0:
            reasons.append(AppraisalReasonCode.THREAT)

        return AppraisalResult(
            novelty=measurement.calibrated_novelty if measurement.valid else None,
            novelty_valid=measurement.valid,
            goal_progress=signals.goal_progress,
            threat=signals.threat,
            controllability=signals.controllability,
            certainty=signals.certainty,
            social_relevance=signals.social_relevance,
            effort_cost=signals.effort_cost,
            reasons=tuple(reasons),
        )

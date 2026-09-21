"""Allostatic emotion-state update logic and structured appraisal updates."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Callable

from kagya.cognition.appraisal import AppraisalResult


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded(value: float, name: str, lower: float, upper: float) -> float:
    result = _number(value, name)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


@dataclass(frozen=True, slots=True)
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    optimal_loss: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", _bounded(self.valence, "valence", -1.0, 1.0))
        object.__setattr__(self, "arousal", _bounded(self.arousal, "arousal", 0.0, 1.0))
        # Keep the legacy scalar finite without imposing a new lower bound on
        # the raw ``update()`` compatibility path.
        object.__setattr__(self, "optimal_loss", _number(self.optimal_loss, "optimal_loss"))


@dataclass(frozen=True, slots=True)
class ValenceContributions:
    goal_progress: float = 0.0
    threat: float = 0.0
    effort_cost: float = 0.0
    controllability: float = 0.0

    def __post_init__(self) -> None:
        bounds = {
            "goal_progress": (-0.7, 0.7),
            "threat": (-0.8, 0.0),
            "effort_cost": (-0.3, 0.0),
            "controllability": (-0.1, 0.1),
        }
        for field, (lower, upper) in bounds.items():
            object.__setattr__(
                self, field, _bounded(getattr(self, field), field, lower, upper)
            )


@dataclass(frozen=True, slots=True)
class ArousalContributions:
    novelty: float = 0.0
    threat: float = 0.0
    effort_cost: float = 0.0
    social_relevance: float = 0.0
    uncertainty: float = 0.0
    low_controllability: float = 0.0

    def __post_init__(self) -> None:
        bounds = {
            "novelty": (0.0, 0.6),
            "threat": (0.0, 0.7),
            "effort_cost": (0.0, 0.3),
            "social_relevance": (0.0, 0.2),
            "uncertainty": (0.0, 0.2),
            "low_controllability": (0.0, 0.2),
        }
        for field, (lower, upper) in bounds.items():
            object.__setattr__(
                self, field, _bounded(getattr(self, field), field, lower, upper)
            )


class EmotionUpdateReasonCode(str, Enum):
    APPRAISAL_APPLIED = "appraisal_applied"
    NOVELTY_OMITTED = "novelty_omitted"
    TIME_RECOVERY = "time_recovery"
    TIMELINE_INITIALIZED = "timeline_initialized"
    NO_MATERIAL_APPRAISAL = "no_material_appraisal"


EmotionUpdateReason = EmotionUpdateReasonCode


@dataclass(frozen=True, slots=True)
class EmotionTemporalState:
    last_update_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.last_update_at is not None:
            _utc_datetime(self.last_update_at)


@dataclass(frozen=True, slots=True)
class EmotionUpdate:
    state: EmotionState
    valence_contributions: ValenceContributions
    arousal_contributions: ArousalContributions
    reasons: tuple[EmotionUpdateReasonCode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, EmotionState):
            raise TypeError("state must be EmotionState")
        if not isinstance(self.valence_contributions, ValenceContributions):
            raise TypeError("valence_contributions must be ValenceContributions")
        if not isinstance(self.arousal_contributions, ArousalContributions):
            raise TypeError("arousal_contributions must be ArousalContributions")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, EmotionUpdateReasonCode) for reason in self.reasons
        ):
            raise TypeError("reasons must be a tuple of EmotionUpdateReasonCode")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must not contain duplicates")


class EmotionEngineAllostasis:
    """Update emotion state from prediction loss or structured appraisal."""

    def __init__(
        self,
        state: EmotionState | None = None,
        adaptation_rate: float = 0.05,
        appraisal_response_rate: float = 0.4,
        resting_valence: float = 0.0,
        resting_arousal: float = 0.0,
        valence_recovery_rate: float = 0.01,
        arousal_recovery_rate: float = 0.02,
        temporal_state: EmotionTemporalState | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.state = state if state is not None else EmotionState()
        # ``adaptation_rate`` retains the legacy finite clamp used by
        # ``update()``; the newly introduced U2 policy values are strict.
        finite_adaptation_rate = _number(adaptation_rate, "adaptation_rate")
        self.adaptation_rate = max(0.0, min(1.0, finite_adaptation_rate))
        self.appraisal_response_rate = _bounded(appraisal_response_rate, "appraisal_response_rate", 0.0, 1.0)
        self.resting_valence = _bounded(resting_valence, "resting_valence", -1.0, 1.0)
        self.resting_arousal = _bounded(resting_arousal, "resting_arousal", 0.0, 1.0)
        self.valence_recovery_rate = _bounded(valence_recovery_rate, "valence_recovery_rate", 0.0, math.inf)
        self.arousal_recovery_rate = _bounded(arousal_recovery_rate, "arousal_recovery_rate", 0.0, math.inf)
        if temporal_state is not None and not isinstance(temporal_state, EmotionTemporalState):
            raise TypeError("temporal_state must be EmotionTemporalState")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.temporal_state = temporal_state if temporal_state is not None else EmotionTemporalState()
        self._clock = clock

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
        self.state = EmotionState(valence=valence, arousal=arousal, optimal_loss=optimal_loss)
        return self.state

    def update_from_appraisal(self, appraisal: AppraisalResult, *, primary_loss: float | None = None) -> EmotionUpdate:
        if not isinstance(appraisal, AppraisalResult):
            raise TypeError("appraisal must be AppraisalResult")
        loss = (
            self.state.optimal_loss
            if primary_loss is None
            else _bounded(primary_loss, "primary_loss", 0.0, math.inf)
        )
        vc = ValenceContributions(
            goal_progress=0.7 * (appraisal.goal_progress or 0.0),
            threat=-0.8 * (appraisal.threat or 0.0),
            effort_cost=-0.3 * (appraisal.effort_cost or 0.0),
            controllability=0.2 * (appraisal.controllability - 0.5) if appraisal.controllability is not None else 0.0,
        )
        ac = ArousalContributions(
            novelty=0.6 * (appraisal.novelty or 0.0) if appraisal.novelty_valid else 0.0,
            threat=0.7 * (appraisal.threat or 0.0),
            effort_cost=0.3 * (appraisal.effort_cost or 0.0),
            social_relevance=0.2 * (appraisal.social_relevance or 0.0),
            uncertainty=0.2 * (1.0 - appraisal.certainty) if appraisal.certainty is not None else 0.0,
            low_controllability=0.2 * (1.0 - appraisal.controllability) if appraisal.controllability is not None else 0.0,
        )
        valence_values = (
            vc.goal_progress,
            vc.threat,
            vc.effort_cost,
            vc.controllability,
        )
        arousal_values = (
            ac.novelty,
            ac.threat,
            ac.effort_cost,
            ac.social_relevance,
            ac.uncertainty,
            ac.low_controllability,
        )
        material = any(value != 0.0 for value in (*valence_values, *arousal_values))
        reasons: list[EmotionUpdateReasonCode] = []
        if not appraisal.novelty_valid:
            reasons.append(EmotionUpdateReasonCode.NOVELTY_OMITTED)
        if material:
            reasons.append(EmotionUpdateReasonCode.APPRAISAL_APPLIED)
            valence_target = _clamp(math.fsum(valence_values), -1.0, 1.0)
            arousal_target = _clamp(math.fsum(arousal_values), 0.0, 1.0)
            valence = _approach(
                self.state.valence, valence_target, self.appraisal_response_rate
            )
            arousal = _approach(
                self.state.arousal, arousal_target, self.appraisal_response_rate
            )
        else:
            reasons.append(EmotionUpdateReasonCode.NO_MATERIAL_APPRAISAL)
            valence, arousal = self.state.valence, self.state.arousal
        optimal_loss = (
            (1.0 - self.adaptation_rate) * self.state.optimal_loss
            + self.adaptation_rate * loss
            if primary_loss is not None
            else self.state.optimal_loss
        )
        new_state = EmotionState(valence=valence, arousal=arousal, optimal_loss=optimal_loss)
        self.state = new_state
        return EmotionUpdate(new_state, vc, ac, tuple(reasons))

    def advance_time(self, elapsed_seconds: float) -> EmotionUpdate:
        elapsed = _bounded(elapsed_seconds, "elapsed_seconds", 0.0, math.inf)
        state = self.state if elapsed == 0.0 else self._recovered_state(elapsed)
        if elapsed != 0.0:
            self.state = state
        return EmotionUpdate(
            state,
            ValenceContributions(),
            ArousalContributions(),
            (EmotionUpdateReasonCode.TIME_RECOVERY,),
        )

    def advance_to(self, now: datetime | None = None) -> EmotionUpdate:
        timestamp = _utc_datetime(self._clock() if now is None else now)
        previous = self.temporal_state.last_update_at
        if previous is None:
            self.temporal_state = EmotionTemporalState(timestamp)
            return EmotionUpdate(self.state, ValenceContributions(), ArousalContributions(), (EmotionUpdateReasonCode.TIMELINE_INITIALIZED,))
        if timestamp < previous:
            raise ValueError("timestamp cannot regress")
        update = self.advance_time((timestamp - previous).total_seconds())
        self.temporal_state = EmotionTemporalState(timestamp)
        return update

    def _recovered_state(self, elapsed: float) -> EmotionState:
        valence_factor = math.exp(-self.valence_recovery_rate * elapsed) if self.valence_recovery_rate else 1.0
        arousal_factor = math.exp(-self.arousal_recovery_rate * elapsed) if self.arousal_recovery_rate else 1.0
        return EmotionState(
            valence=self.resting_valence + (self.state.valence - self.resting_valence) * valence_factor,
            arousal=self.resting_arousal + (self.state.arousal - self.resting_arousal) * arousal_factor,
            optimal_loss=self.state.optimal_loss,
        )


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None or offset.total_seconds() != 0.0:
        raise ValueError("timestamp must be timezone-aware UTC")
    return value


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


def _approach(current: float, target: float, response_rate: float) -> float:
    if response_rate == 0.0:
        return current
    if response_rate == 1.0:
        return target
    return current + (target - current) * response_rate

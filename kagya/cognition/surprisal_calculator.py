"""Surprisal calculation and bounded, privacy-preserving calibration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
import sys
from typing import Final, Iterable

from kagya.models import ModelProvider


MODEL_KEY_DOMAIN: Final[bytes] = b"PROJECT-KAGYA:R10:CALIBRATION-MODEL:V1\0"
_MODEL_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"model\.[0-9a-f]{64}")


def _is_model_key(value: object) -> bool:
    return isinstance(value, str) and _MODEL_KEY_PATTERN.fullmatch(value) is not None


def _validate_identity_part(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError(f"{name} must be a non-empty identity token")
    return value


def model_key(provider_name: str, model_id: str) -> str:
    """Derive an opaque stable key from configured provider/model identity."""

    provider = _validate_identity_part(provider_name, "provider_name")
    model = _validate_identity_part(model_id, "model_id")
    canonical_identity = json.dumps(
        [provider, model], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(MODEL_KEY_DOMAIN + canonical_identity).hexdigest()
    return f"model.{digest}"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _safe_mean(previous: float, sample: float, count: int) -> float:
    """Compute a finite weighted mean without overflowing a difference."""

    weight = (count - 1) / count
    try:
        result = math.fsum((previous * weight, sample / count))
    except OverflowError:
        result = math.copysign(sys.float_info.max, previous + sample)
    if math.isfinite(result):
        return result
    return math.copysign(sys.float_info.max, previous + sample)


def _safe_m2(previous: float, sample: float, previous_mean: float, mean: float) -> float:
    """Keep Welford's non-negative variance accumulator finite at float limits."""

    delta = sample - previous_mean
    correction = sample - mean
    try:
        increment = delta * correction
        result = previous + increment
    except OverflowError:
        return sys.float_info.max
    if not math.isfinite(result) or result < 0.0:
        return sys.float_info.max
    return min(result, sys.float_info.max)


def _validate_calibration_values(
    model_key_value: object,
    count: object,
    mean: object,
    m2: object,
) -> tuple[float, float]:
    if not _is_model_key(model_key_value):
        raise ValueError("model_key must be an opaque model key")
    if type(count) is not int or count < 0:
        raise TypeError("count must be a non-negative integer")
    mean_value = _finite_number(mean, "mean")
    m2_value = _finite_number(m2, "m2")
    if m2_value < 0.0:
        raise ValueError("m2 must be non-negative")
    if count == 0 and (mean_value != 0.0 or m2_value != 0.0):
        raise ValueError("an empty calibration entry must be zero")
    return mean_value, m2_value


@dataclass(frozen=True, slots=True)
class CalibrationEntry:
    """One exact Welford state for an approved model key."""

    model_key: str
    count: int
    mean: float
    m2: float

    def __post_init__(self) -> None:
        mean, m2 = _validate_calibration_values(
            self.model_key, self.count, self.mean, self.m2
        )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "m2", m2)


class LossInvalidReason(str, Enum):
    EMPTY_TARGET = "empty_target"
    PROVIDER_ERROR = "provider_error"
    NON_FINITE_LOSS = "non_finite_loss"


@dataclass(frozen=True, slots=True)
class LossMeasurement:
    """Typed provider-loss evidence with explicit validity."""

    model_key: str
    raw_loss: float | None
    valid: bool
    invalid_reason: LossInvalidReason | None
    calibrated_novelty: float | None

    def __post_init__(self) -> None:
        if not _is_model_key(self.model_key):
            raise ValueError("model_key must be an opaque model key")
        if type(self.valid) is not bool:
            raise TypeError("valid must be bool")
        if self.valid:
            if self.invalid_reason is not None:
                raise ValueError("valid measurements cannot have an invalid reason")
            if self.raw_loss is None or self.calibrated_novelty is None:
                raise ValueError("valid measurements require loss and novelty")
            raw_loss = _finite_number(self.raw_loss, "raw_loss")
            novelty = _finite_number(self.calibrated_novelty, "calibrated_novelty")
            if not 0.0 <= novelty <= 1.0:
                raise ValueError("calibrated_novelty must be in [0, 1]")
            object.__setattr__(self, "raw_loss", raw_loss)
            object.__setattr__(self, "calibrated_novelty", novelty)
        else:
            if self.invalid_reason is None or not isinstance(
                self.invalid_reason, LossInvalidReason
            ):
                raise ValueError("invalid measurements require a closed reason")
            if self.raw_loss is not None or self.calibrated_novelty is not None:
                raise ValueError("invalid measurements cannot carry numeric evidence")


class LossCalibration:
    """Bounded Welford statistics with exact, atomic snapshot restoration."""

    def __init__(
        self,
        approved_keys: Iterable[str],
        *,
        initial_baseline: float,
        initial_scale: float,
        minimum_scale: float,
    ) -> None:
        if isinstance(approved_keys, (str, bytes)):
            raise TypeError("approved_keys must be an iterable of model keys")
        try:
            keys = tuple(approved_keys)
        except TypeError as error:
            raise TypeError("approved_keys must be an iterable of model keys") from error
        canonical_keys = tuple(sorted(set(keys)))
        if not canonical_keys or len(canonical_keys) > 64:
            raise ValueError("approved keys must be non-empty and bounded")
        if any(not _is_model_key(key) for key in canonical_keys):
            raise ValueError("approved keys must be opaque model keys")
        baseline = _finite_number(initial_baseline, "initial_baseline")
        scale = _finite_number(initial_scale, "initial_scale")
        minimum = _finite_number(minimum_scale, "minimum_scale")
        if scale <= 0.0 or minimum <= 0.0:
            raise ValueError("calibration scales must be positive")
        self._approved = frozenset(canonical_keys)
        self._initial_baseline = baseline
        self._initial_scale = scale
        self._minimum_scale = minimum
        self._entries: dict[str, CalibrationEntry] = {}

    @property
    def approved_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._approved))

    def export(self) -> tuple[CalibrationEntry, ...]:
        """Return the immutable canonical ordered calibration state."""

        return tuple(self._entries[key] for key in sorted(self._entries))

    def restore_exact(self, state: tuple[CalibrationEntry, ...]) -> None:
        """Atomically replace state after validating the complete snapshot."""

        if type(state) is not tuple:
            raise TypeError("calibration state must be a tuple")
        candidate: dict[str, CalibrationEntry] = {}
        previous_key: str | None = None
        for entry in state:
            if not isinstance(entry, CalibrationEntry):
                raise TypeError("calibration state contains an invalid entry")
            _validate_calibration_values(
                entry.model_key, entry.count, entry.mean, entry.m2
            )
            if entry.model_key not in self._approved:
                raise ValueError("calibration entry uses an unapproved key")
            if previous_key is not None and entry.model_key <= previous_key:
                raise ValueError("calibration state must be strictly ordered")
            previous_key = entry.model_key
            candidate[entry.model_key] = entry
        self._entries = candidate

    def sample(self, key: str, loss: float) -> float:
        """Score one finite loss using pre-sample state, then update Welford state."""

        if not isinstance(key, str) or key not in self._approved:
            raise ValueError("model_key is not approved")
        sample = _finite_number(loss, "loss")
        old = self._entries.get(key, CalibrationEntry(key, 0, 0.0, 0.0))
        baseline = self._initial_baseline if old.count == 0 else old.mean
        if old.count > 1:
            scale = math.sqrt(old.m2 / (old.count - 1))
        else:
            scale = self._initial_scale
        scale = max(scale, self._minimum_scale)
        novelty = _sigmoid((sample - baseline) / scale)

        count = old.count + 1
        if count == 1:
            mean = sample
            m2 = 0.0
        else:
            mean = _safe_mean(old.mean, sample, count)
            m2 = _safe_m2(old.m2, sample, old.mean, mean)
        candidate = CalibrationEntry(key, count, mean, m2)
        self._entries[key] = candidate
        return novelty


class SurprisalCalculator:
    """Thin wrapper over provider loss for new target text."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def calculate(self, context_text: str, target_text: str) -> float:
        return self.provider.calculate_loss(context_text, target_text)

    def measure(
        self,
        context_text: str,
        target_text: str,
        *,
        model_key: str,
        calibration: LossCalibration,
    ) -> LossMeasurement:
        """Return typed loss evidence without leaking provider failures."""

        if not isinstance(calibration, LossCalibration):
            raise TypeError("calibration must be LossCalibration")
        if not isinstance(model_key, str) or model_key not in calibration.approved_keys:
            raise ValueError("model_key is not approved")
        if not isinstance(target_text, str):
            raise TypeError("target_text must be a string")
        if target_text == "":
            return LossMeasurement(
                model_key, None, False, LossInvalidReason.EMPTY_TARGET, None
            )
        try:
            raw_loss = self.provider.calculate_loss(context_text, target_text)
            if isinstance(raw_loss, bool) or not isinstance(raw_loss, (int, float)):
                raise TypeError("provider loss must be numeric")
            loss = float(raw_loss)
        except Exception:
            return LossMeasurement(
                model_key, None, False, LossInvalidReason.PROVIDER_ERROR, None
            )
        if not math.isfinite(loss):
            return LossMeasurement(
                model_key, None, False, LossInvalidReason.NON_FINITE_LOSS, None
            )
        novelty = calibration.sample(model_key, loss)
        return LossMeasurement(model_key, loss, True, None, novelty)

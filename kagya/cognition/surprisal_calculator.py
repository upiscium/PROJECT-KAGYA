"""Surprisal calculation over configured model providers."""

from dataclasses import dataclass
import math

from kagya.cognition.appraisal import LossMeasurement, bounded_sigmoid
from kagya.models import ModelProvider


@dataclass
class CalibrationState:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0


class SurprisalCalculator:
    """Thin wrapper over provider loss for new target text."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        initial_baseline: float = 1.0,
        initial_scale: float = 0.5,
        minimum_scale: float = 0.1,
    ) -> None:
        self.provider = provider
        self.initial_baseline = initial_baseline
        self.initial_scale = initial_scale
        self.minimum_scale = minimum_scale
        self.history: dict[str, CalibrationState] = {}

    def calculate(self, context_text: str, target_text: str) -> float:
        return self.provider.calculate_loss(context_text, target_text)

    def measure(self, context_text: str, target_text: str, *, model_key: str) -> LossMeasurement:
        if not target_text:
            return LossMeasurement(None, None, 0, model_key, False, "empty_target", None)
        try:
            loss = float(self.provider.calculate_loss(context_text, target_text))
        except Exception:
            return LossMeasurement(None, None, None, model_key, False, "provider_error", None)
        if not math.isfinite(loss):
            return LossMeasurement(None, None, None, model_key, False, "non_finite_loss", None)
        state = self.history.setdefault(model_key, CalibrationState())
        baseline = state.mean if state.count else self.initial_baseline
        variance = state.m2 / (state.count - 1) if state.count > 1 else self.initial_scale**2
        scale = max(self.minimum_scale, math.sqrt(max(0.0, variance)))
        novelty = bounded_sigmoid((loss - baseline) / scale)
        state.count += 1
        delta = loss - state.mean
        state.mean += delta / state.count
        state.m2 += delta * (loss - state.mean)
        return LossMeasurement(
            raw_loss=loss,
            mean_token_loss=loss,
            target_token_count=max(1, len(target_text.encode("utf-8"))),
            model_key=model_key,
            valid=True,
            invalid_reason=None,
            calibrated_novelty=novelty,
        )

    def export_history(self) -> dict[str, dict[str, float | int]]:
        return {
            key: {"count": value.count, "mean": value.mean, "m2": value.m2}
            for key, value in self.history.items()
        }

    def restore_history(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self.history = {}
            return
        restored: dict[str, CalibrationState] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            try:
                state = CalibrationState(
                    count=int(value["count"]),
                    mean=float(value["mean"]),
                    m2=float(value["m2"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if state.count >= 0 and math.isfinite(state.mean) and math.isfinite(state.m2):
                restored[key] = state
        self.history = restored

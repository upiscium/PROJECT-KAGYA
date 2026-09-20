from dataclasses import FrozenInstanceError
import math

import pytest

from kagya.cognition import (
    CalibrationEntry,
    LossCalibration,
    LossInvalidReason,
    LossMeasurement,
    SurprisalCalculator,
    model_key,
)
from kagya.models import DummyProvider


class RecordingProvider(DummyProvider):
    def __init__(self, value: object = DummyProvider.loss_value) -> None:
        self.calls: list[tuple[str, str]] = []
        self.value = value

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        self.calls.append((context_text, target_text))
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value  # type: ignore[return-value]


def key(provider: str = "dummy", model: str = "model") -> str:
    return model_key(provider, model)


def calibration(*keys: str, baseline: float = 1.0, scale: float = 1.0, minimum: float = 0.01) -> LossCalibration:
    return LossCalibration(
        keys,
        initial_baseline=baseline,
        initial_scale=scale,
        minimum_scale=minimum,
    )


def test_surprisal_delegates_to_provider_with_dummy_loss() -> None:
    provider = RecordingProvider()
    calculator = SurprisalCalculator(provider)

    loss = calculator.calculate("prior context", "new user text")

    assert loss == DummyProvider.loss_value
    assert provider.calls == [("prior context", "new user text")]


def test_surprisal_empty_target_raises_through_provider() -> None:
    calculator = SurprisalCalculator(DummyProvider())

    with pytest.raises(ValueError, match="target_text"):
        calculator.calculate("prior context", "")


def test_surprisal_keeps_long_context_separate_from_target() -> None:
    provider = RecordingProvider()
    calculator = SurprisalCalculator(provider)
    context = "old context " * 100
    target = "new target only"

    calculator.calculate(context, target)

    assert provider.calls[0] == (context, target)


def test_model_key_is_stable_opaque_and_distinguishes_identity() -> None:
    first = model_key("transformers", "org/model")
    assert first == model_key("transformers", "org/model")
    assert first.startswith("model.")
    assert len(first) == len("model.") + 64
    assert first[6:] == first[6:].lower()
    assert "transformers" not in first
    assert "org/model" not in first
    assert first != model_key("dummy", "org/model")
    assert first != model_key("transformers", "other/model")


@pytest.mark.parametrize(
    ("provider_name", "model_id"),
    [
        ("", "model"),
        ("provider", ""),
        (" ", "model"),
        ("provider", " model"),
        ("provider\n", "model"),
        ("provider", "model\x00"),
        (None, "model"),
        ("provider", None),
    ],
)
def test_model_key_rejects_invalid_identity(
    provider_name: object, model_id: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        model_key(provider_name, model_id)  # type: ignore[arg-type]


def test_duplicate_approved_keys_canonicalize_and_state_is_sorted() -> None:
    first, second = key("dummy", "a"), key("dummy", "b")
    state = calibration(second, first, first)

    assert state.approved_keys == tuple(sorted((first, second)))
    state.sample(second, 2.0)
    state.sample(first, 1.0)

    assert tuple(entry.model_key for entry in state.export()) == tuple(
        sorted((first, second))
    )


def test_first_sample_uses_explicit_warm_baseline_and_scale() -> None:
    model = key()
    state = calibration(model, baseline=2.0, scale=4.0, minimum=0.1)

    novelty = state.sample(model, 6.0)

    assert novelty == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert state.export() == (CalibrationEntry(model, 1, 6.0, 0.0),)


def test_welford_state_is_exact_after_multiple_samples() -> None:
    model = key()
    state = calibration(model, baseline=0.0, scale=1.0, minimum=0.01)

    for value in (1.0, 2.0, 3.0):
        state.sample(model, value)

    assert state.export() == (CalibrationEntry(model, 3, 2.0, 2.0),)


def test_pre_sample_sample_standard_deviation_and_minimum_scale() -> None:
    model = key()
    state = calibration(model, baseline=0.0, scale=2.0, minimum=2.0)
    state.sample(model, 1.0)
    state.sample(model, 1.0)

    # count == 2 has zero sample variance, so configured minimum_scale wins.
    novelty = state.sample(model, 3.0)

    assert novelty == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_novelty_is_bounded_for_extreme_finite_losses() -> None:
    model = key()
    for value in (-1e308, 1e308):
        state = calibration(model)
        novelty = state.sample(model, value)
        assert math.isfinite(novelty)
        assert 0.0 <= novelty <= 1.0


def test_cross_extreme_finite_losses_keep_calibration_bounded() -> None:
    model = key()
    state = calibration(model)

    state.sample(model, 1e308)
    novelty = state.sample(model, -1e308)

    entry = state.export()[0]
    assert math.isfinite(novelty)
    assert 0.0 <= novelty <= 1.0
    assert math.isfinite(entry.mean)
    assert math.isfinite(entry.m2)
    assert entry.m2 >= 0.0


def test_measure_handles_cross_extreme_finite_provider_losses() -> None:
    model = key()
    provider = RecordingProvider(1e308)
    state = calibration(model)
    calculator = SurprisalCalculator(provider)

    first = calculator.measure("context", "first", model_key=model, calibration=state)
    provider.value = -1e308
    second = calculator.measure("context", "second", model_key=model, calibration=state)

    assert first.valid and second.valid
    assert state.export()[0].count == 2


def test_models_have_independent_calibration_state() -> None:
    first, second = key("dummy", "a"), key("dummy", "b")
    state = calibration(first, second, baseline=0.0, scale=1.0, minimum=0.01)

    state.sample(first, 2.0)

    assert state.export() == (CalibrationEntry(first, 1, 2.0, 0.0),)
    assert second in state.approved_keys


def test_exact_restore_round_trip_replaces_state() -> None:
    model = key()
    original = calibration(model, baseline=0.0, scale=1.0, minimum=0.1)
    original.sample(model, 1.0)
    original.sample(model, 3.0)
    exported = original.export()
    restored = calibration(model, baseline=0.0, scale=1.0, minimum=0.1)

    restored.restore_exact(exported)

    assert restored.export() == exported


def _forged_entry(model: str, count: object, mean: object, m2: object) -> CalibrationEntry:
    entry = object.__new__(CalibrationEntry)
    object.__setattr__(entry, "model_key", model)
    object.__setattr__(entry, "count", count)
    object.__setattr__(entry, "mean", mean)
    object.__setattr__(entry, "m2", m2)
    return entry


@pytest.mark.parametrize(
    "state_factory",
    [
        lambda model, other: [_forged_entry(model, 1, 1.0, 0.0)],
        lambda model, other: (
            CalibrationEntry(model, 1, 1.0, 0.0),
            CalibrationEntry(model, 1, 1.0, 0.0),
        ),
        lambda model, other: (CalibrationEntry(other, 1, 1.0, 0.0),),
        lambda model, other: (_forged_entry(model, -1, 1.0, 0.0),),
        lambda model, other: (_forged_entry(model, True, 1.0, 0.0),),
        lambda model, other: (_forged_entry(model, 1, math.nan, 0.0),),
        lambda model, other: (_forged_entry(model, 1, math.inf, 0.0),),
        lambda model, other: (_forged_entry(model, 1, 1.0, math.nan),),
        lambda model, other: (_forged_entry(model, 1, 1.0, math.inf),),
        lambda model, other: (_forged_entry(model, 1, 1.0, -1.0),),
        lambda model, other: (_forged_entry(model, 1, 1.0, 1.0),),
        lambda model, other: (
            tuple(
                sorted(
                    (
                        CalibrationEntry(other, 1, 1.0, 0.0),
                        CalibrationEntry(model, 1, 1.0, 0.0),
                    ),
                    key=lambda entry: entry.model_key,
                    reverse=True,
                )
            )
        ),
    ],
)
def test_malformed_restore_fails_atomically(state_factory) -> None:
    model, other = key(), key("dummy", "other")
    state = calibration(model)
    state.sample(model, 2.0)
    before = state.export()

    malformed = state_factory(model, other)
    with pytest.raises((TypeError, ValueError)):
        state.restore_exact(malformed)
    assert state.export() == before

    with pytest.raises(TypeError):
        state.restore_exact(list(before))  # type: ignore[arg-type]


def test_calibration_entries_are_immutable() -> None:
    entry = CalibrationEntry(key(), 1, 1.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        entry.count = 2  # type: ignore[misc]


def test_single_sample_calibration_requires_zero_m2() -> None:
    with pytest.raises(ValueError, match="m2 must be zero"):
        CalibrationEntry(key(), 1, 1.0, 1.0)


def test_unapproved_key_fails_before_provider_call() -> None:
    provider = RecordingProvider()
    calculator = SurprisalCalculator(provider)
    approved = key()
    state = calibration(approved)

    with pytest.raises(ValueError):
        calculator.measure("context", "target", model_key="model." + "0" * 64, calibration=state)

    assert provider.calls == []
    assert state.export() == ()


class FailingProvider(DummyProvider):
    def calculate_loss(self, context_text: str, target_text: str) -> float:
        raise RuntimeError("private provider exception sentinel")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_losses_are_invalid_without_mutation(value: float) -> None:
    provider = RecordingProvider(value)
    calculator = SurprisalCalculator(provider)
    model = key()
    state = calibration(model)
    before = state.export()

    result = calculator.measure("context", "target", model_key=model, calibration=state)

    assert not result.valid
    assert result.raw_loss is None
    assert result.calibrated_novelty is None
    assert result.invalid_reason is LossInvalidReason.NON_FINITE_LOSS
    assert state.export() == before


def test_provider_exception_becomes_bounded_private_free_evidence() -> None:
    model = key()
    state = calibration(model)
    result = SurprisalCalculator(FailingProvider()).measure(
        "private context", "private target", model_key=model, calibration=state
    )

    assert result == LossMeasurement(
        model, None, False, LossInvalidReason.PROVIDER_ERROR, None
    )
    assert "private provider exception sentinel" not in repr(result)
    assert state.export() == ()


def test_empty_target_is_invalid_without_calling_provider() -> None:
    provider = RecordingProvider()
    model = key()
    state = calibration(model)

    result = SurprisalCalculator(provider).measure(
        "context", "", model_key=model, calibration=state
    )

    assert result.invalid_reason is LossInvalidReason.EMPTY_TARGET
    assert not result.valid
    assert provider.calls == []
    assert state.export() == ()


def test_valid_measurement_updates_calibration_and_is_bounded() -> None:
    model = key()
    state = calibration(model)
    result = SurprisalCalculator(RecordingProvider()).measure(
        "context", "target", model_key=model, calibration=state
    )

    assert result.valid
    assert result.raw_loss == DummyProvider.loss_value
    assert result.invalid_reason is None
    assert result.calibrated_novelty is not None
    assert 0.0 <= result.calibrated_novelty <= 1.0
    assert state.export()[0].count == 1


def test_loss_measurement_invariants_are_strict_and_immutable() -> None:
    model = key()
    with pytest.raises(ValueError):
        LossMeasurement(model, None, True, None, None)
    with pytest.raises(ValueError):
        LossMeasurement(model, 1.0, False, LossInvalidReason.PROVIDER_ERROR, 0.5)
    with pytest.raises(ValueError):
        LossMeasurement(model, 1.0, True, None, 1.5)
    measurement = LossMeasurement(model, 1.0, True, None, 0.5)
    with pytest.raises(FrozenInstanceError):
        measurement.raw_loss = 2.0  # type: ignore[misc]

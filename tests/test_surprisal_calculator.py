import pytest

from kagya.cognition import SurprisalCalculator
from kagya.models import DummyProvider


class RecordingProvider(DummyProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        self.calls.append((context_text, target_text))
        return super().calculate_loss(context_text, target_text)


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

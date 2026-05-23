"""Surprisal calculation over configured model providers."""

from kagya.models import ModelProvider


class SurprisalCalculator:
    """Thin wrapper over provider loss for new target text."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def calculate(self, context_text: str, target_text: str) -> float:
        return self.provider.calculate_loss(context_text, target_text)

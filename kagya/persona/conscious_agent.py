"""Provider-agnostic conscious agent wrapper."""

from kagya.models import ModelProvider


class ConsciousAgent:
    """Generate raw model responses without postprocessing."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def generate(self, prompt: str) -> str:
        return self.provider.generate(prompt)

"""Provider-agnostic conscious agent wrapper."""

from kagya.models import ModelProvider


class ConsciousAgent:
    """Generate raw model responses without postprocessing."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def generate(
        self, prompt: str, *, attachments: list[dict[str, object]] | None = None
    ) -> str:
        generate_with_attachments = getattr(self.provider, "generate_with_attachments", None)
        if attachments and callable(generate_with_attachments):
            return str(generate_with_attachments(prompt, attachments))
        return self.provider.generate(prompt)

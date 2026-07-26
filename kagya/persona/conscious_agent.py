"""Provider-agnostic conscious agent wrapper."""

from kagya.models import ModelProvider
from kagya.runtime.cancellation import current_cancellation_token


class ConsciousAgent:
    """Generate raw model responses without postprocessing."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def generate(
        self, prompt: str, *, attachments: list[dict[str, object]] | None = None
    ) -> str:
        token = current_cancellation_token()
        generate_with_attachments = getattr(self.provider, "generate_with_attachments", None)
        if attachments and callable(generate_with_attachments):
            return str(generate_with_attachments(prompt, attachments))
        stream_generate = getattr(self.provider, "stream_generate", None)
        if callable(stream_generate):
            return "".join(stream_generate(prompt, token))
        if token is not None:
            token.raise_if_canceled()
        return self.provider.generate(prompt)

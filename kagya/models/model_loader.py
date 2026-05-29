"""Factory for configured model providers."""

from kagya.config import Settings, get_settings
from kagya.models.base import ModelProvider
from kagya.models.dummy_provider import DummyProvider
from kagya.models.transformers_provider import TransformersProvider


def load_model_provider(settings: Settings | None = None) -> ModelProvider:
    """Load the configured model provider."""

    app_settings = settings or get_settings()
    provider_name = app_settings.model.provider.lower()
    if provider_name == "dummy":
        return DummyProvider()
    if provider_name == "transformers":
        return TransformersProvider(app_settings)
    raise ValueError(f"Unsupported model provider: {app_settings.model.provider}")

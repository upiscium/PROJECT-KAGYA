"""Factory for configured model providers."""

from pathlib import Path

from kagya.config import Settings, get_settings
from kagya.models.base import ModelProvider
from kagya.models.dummy_provider import DummyProvider
from kagya.models.transformers_provider import TransformersProvider


def load_model_provider(settings: Settings | None = None, adapter_path: str | Path | None = None) -> ModelProvider:
    """Load the configured model provider."""

    app_settings = settings or get_settings()
    provider_name = app_settings.model.provider.lower()
    if provider_name == "dummy":
        return DummyProvider()
    if provider_name == "transformers":
        return TransformersProvider(app_settings, adapter_path=adapter_path)
    raise ValueError(f"Unsupported model provider: {app_settings.model.provider}")

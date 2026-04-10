from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from project_kagya.quantized_model_loader import (
    QuantizedModelLoader,
    QuantizationConfig,
)


class DummyModel:
    def __init__(self) -> None:
        self.named_parameters = lambda: []


def test_build_quantization_config_returns_defaults() -> None:
    loader = QuantizedModelLoader()

    config = loader.build_quantization_config()

    assert config == QuantizationConfig()


def test_load_4bit_model_rejects_empty_model_name() -> None:
    loader = QuantizedModelLoader()

    with pytest.raises(ValueError, match="model_name_or_path must not be empty"):
        loader.load_4bit_model("")


def test_prepare_for_training_returns_model_without_peft_dependency() -> None:
    loader = QuantizedModelLoader()
    model = DummyModel()

    prepared = loader.prepare_for_training(model)

    assert prepared is model


def test_load_4bit_model_uses_transformers_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = QuantizedModelLoader()
    captured: dict[str, Any] = {}

    transformers = ModuleType("transformers")

    class FakeQuantizationConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(
            model_name_or_path: str,
            device_map: str | dict[str, Any] | None = None,
            quantization_config: Any | None = None,
        ) -> dict[str, Any]:
            captured["model_name_or_path"] = model_name_or_path
            captured["device_map"] = device_map
            captured["quantization_config"] = quantization_config
            return {"model_name_or_path": model_name_or_path}

    transformers.AutoModelForCausalLM = FakeModelLoader  # type: ignore[attr-defined]
    transformers.BitsAndBytesConfig = FakeQuantizationConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    result = loader.load_4bit_model("test-model", device_map="auto")

    assert result == {"model_name_or_path": "test-model"}
    assert captured["model_name_or_path"] == "test-model"
    assert captured["device_map"] == "auto"
    assert captured["quantization_config"].kwargs["load_in_4bit"] is True

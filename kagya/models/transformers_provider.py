"""Hugging Face Transformers-backed model provider."""

from pathlib import Path
from typing import Any
import json

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from kagya.config import Settings


LOADABLE_ADAPTER_STATES = {"trial_active", "approved", "active"}


class TransformersProvider:
    """Provider backed by the configured Hugging Face Transformers model."""

    def __init__(
        self,
        settings: Settings,
        adapter_path: str | Path | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model_id = settings.model.primary_id
        self.processor = processor or AutoProcessor.from_pretrained(self.model_id)
        self.model = model or self._load_model(self.model_id)
        if adapter_path is not None:
            self.attach_adapter(adapter_path)

    def generate(self, prompt: str) -> str:
        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = self._move_inputs_to_model_device(inputs)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.settings.generation.max_new_tokens,
            temperature=self.settings.generation.temperature,
            top_p=self.settings.generation.top_p,
            do_sample=self.settings.generation.do_sample,
        )
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        if not target_text:
            raise ValueError("target_text must not be empty")

        self.model.eval()
        full_inputs = self._tokenize(context_text + target_text)
        context_inputs = self._tokenize(context_text) if context_text else None
        labels = full_inputs["input_ids"].clone()
        context_length = 0 if context_inputs is None else context_inputs["input_ids"].shape[1]
        labels[:, :context_length] = -100
        full_inputs = self._move_inputs_to_model_device(full_inputs)
        labels = labels.to(full_inputs["input_ids"].device)

        with torch.no_grad():
            outputs = self.model(**full_inputs, labels=labels)
        return float(outputs.loss.detach().cpu().item())

    def get_model(self) -> Any:
        return self.model

    def get_processor(self) -> Any:
        return self.processor

    def attach_adapter(self, adapter_path: str | Path) -> None:
        if not is_registry_approved_adapter(self.settings, adapter_path):
            raise ValueError("Adapter path is not approved by the adapter registry")
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))

    def _load_model(self, model_id: str) -> Any:
        load_kwargs: dict[str, Any] = {}
        if self.settings.model.device == "auto":
            load_kwargs["device_map"] = "auto"
        if self.settings.model.dtype != "auto":
            load_kwargs["torch_dtype"] = getattr(torch, self.settings.model.dtype)
        if self.settings.model.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        return AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.processor(text=text, return_tensors="pt")
        return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}

    def _move_inputs_to_model_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = getattr(self.model, "device", None)
        if device is None:
            return inputs
        return {key: value.to(device) for key, value in inputs.items()}


def is_registry_approved_adapter(settings: Settings, adapter_path: str | Path) -> bool:
    """Return whether an adapter path is registered in a loadable state."""

    registry_path = settings.adapter_registry.path
    if not registry_path.exists():
        return False

    requested_path = Path(adapter_path).expanduser().resolve()
    with registry_path.open("r", encoding="utf-8") as registry_file:
        registry_data = json.load(registry_file)
    for entry in _iter_registry_entries(registry_data):
        path = entry.get("path")
        state = entry.get("state")
        if path is None or state not in LOADABLE_ADAPTER_STATES:
            continue
        if Path(path).expanduser().resolve() == requested_path:
            return True
    return False


def _iter_registry_entries(registry_data: Any) -> list[dict[str, Any]]:
    if isinstance(registry_data, list):
        return [entry for entry in registry_data if isinstance(entry, dict)]
    if isinstance(registry_data, dict):
        adapters = registry_data.get("adapters")
        if isinstance(adapters, list):
            return [entry for entry in adapters if isinstance(entry, dict)]
        return [
            {"path": path, "state": state}
            for path, state in registry_data.items()
            if isinstance(path, str) and isinstance(state, str)
        ]
    return []

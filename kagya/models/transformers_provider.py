"""Hugging Face Transformers-backed model provider."""

from pathlib import Path
from typing import Any
import json

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from kagya.config import Settings
from kagya.attachments import ProcessedImageAttachment, validate_image_attachments


LOADABLE_ADAPTER_STATES = {"trial_active", "approved", "active"}


class TransformersProvider:
    """Provider backed by the configured Hugging Face Transformers model."""

    supports_multimodal_attachments = True

    def __init__(
        self,
        settings: Settings,
        adapter_path: str | Path | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model_id = settings.model.primary_id
        self.fallback_model_id = settings.model.fallback_id
        self.adapter_path = adapter_path
        self.processor = processor
        self.model = model
        self._fallback_processor: Any | None = None
        self._fallback_model: Any | None = None
        self._adapter_attached = False
        self.last_model_id = self.model_id
        self.last_fallback_used = False
        if model is not None and adapter_path is not None:
            self.attach_adapter(adapter_path)

    def generate(self, prompt: str) -> str:
        self.last_model_id = self.model_id
        self.last_fallback_used = False
        try:
            return self._generate_with(
                prompt, self._get_primary_model(), self._get_primary_processor()
            )
        except Exception:
            return self.generate_fallback(prompt)

    def generate_with_attachments(
        self, prompt: str, attachments: list[dict[str, object]]
    ) -> str:
        self.last_model_id = self.model_id
        self.last_fallback_used = False
        processed = validate_image_attachments(attachments)
        try:
            return self._generate_with(
                prompt,
                self._get_primary_model(),
                self._get_primary_processor(),
                image_attachments=processed,
            )
        except Exception:
            return self.generate_fallback(prompt)

    def generate_fallback(self, prompt: str) -> str:
        self.last_model_id = self.fallback_model_id
        self.last_fallback_used = True
        return self._generate_with(
            prompt, self._get_fallback_model(), self._get_fallback_processor()
        )

    def _generate_with(
        self,
        prompt: str,
        model: Any,
        processor: Any,
        *,
        image_attachments: list[ProcessedImageAttachment] | None = None,
    ) -> str:
        rendered = self._render_generation_prompt(
            prompt, processor, image_attachments=image_attachments or []
        )
        images = [attachment.image for attachment in image_attachments or []]
        try:
            inputs = processor(text=rendered, images=images, return_tensors="pt") if images else processor(text=rendered, return_tensors="pt")
        except TypeError:
            inputs = processor(text=rendered, return_tensors="pt")
        inputs = self._move_inputs_to_model_device(inputs, model)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.generation.max_new_tokens,
            "do_sample": self.settings.generation.do_sample,
        }
        if self.settings.generation.do_sample:
            generation_kwargs["temperature"] = self.settings.generation.temperature
            generation_kwargs["top_p"] = self.settings.generation.top_p
        output_ids = model.generate(**inputs, **generation_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        return processor.decode(generated_ids, skip_special_tokens=True)

    def _render_generation_prompt(
        self,
        prompt: str,
        processor: Any,
        *,
        image_attachments: list[ProcessedImageAttachment] | None = None,
    ) -> str:
        apply_chat_template = getattr(processor, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return prompt
        content: str | list[dict[str, Any]] = _strip_assistant_marker(prompt)
        if image_attachments:
            content = [{"type": "text", "text": _strip_assistant_marker(prompt)}]
            content.extend({"type": "image"} for _ in image_attachments)
        try:
            rendered = apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            return prompt
        return rendered if isinstance(rendered, str) else prompt

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        if not target_text:
            raise ValueError("target_text must not be empty")

        model = self._get_primary_model()
        model.eval()
        full_inputs = self._tokenize(context_text + target_text)
        context_inputs = self._tokenize(context_text) if context_text else None
        labels = full_inputs["input_ids"].clone()
        context_length = (
            0 if context_inputs is None else context_inputs["input_ids"].shape[1]
        )
        labels[:, :context_length] = -100
        full_inputs = self._move_inputs_to_model_device(full_inputs, model)
        labels = labels.to(full_inputs["input_ids"].device)

        with torch.no_grad():
            outputs = model(**full_inputs, labels=labels)
        return float(outputs.loss.detach().cpu().item())

    def get_model(self) -> Any:
        return self._get_primary_model()

    def get_processor(self) -> Any:
        return self._get_primary_processor()

    def attach_adapter(self, adapter_path: str | Path) -> None:
        if not is_registry_approved_adapter(self.settings, adapter_path):
            raise ValueError("Adapter path is not approved by the adapter registry")
        if self.model is None:
            self.model = self._load_model(self.model_id)
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self._adapter_attached = True

    def _get_primary_processor(self) -> Any:
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self.processor

    def _get_primary_model(self) -> Any:
        if self.model is None:
            self.model = self._load_model(self.model_id)
            self._adapter_attached = False
        if self.adapter_path is not None and not self._adapter_attached:
            self.attach_adapter(self.adapter_path)
        return self.model

    def _get_fallback_processor(self) -> Any:
        if self._fallback_processor is None:
            self._fallback_processor = AutoProcessor.from_pretrained(
                self.fallback_model_id
            )
        return self._fallback_processor

    def _get_fallback_model(self) -> Any:
        if self._fallback_model is None:
            self._fallback_model = self._load_model(self.fallback_model_id)
        return self._fallback_model

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
        encoded = self._get_primary_processor()(text=text, return_tensors="pt")
        return {
            key: value
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    def _move_inputs_to_model_device(
        self, inputs: dict[str, torch.Tensor], model: Any
    ) -> dict[str, torch.Tensor]:
        device = getattr(model, "device", None)
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


def _strip_assistant_marker(prompt: str) -> str:
    marker = "\nAssistant:"
    if prompt.endswith(marker):
        return prompt[: -len(marker)].rstrip()
    return prompt

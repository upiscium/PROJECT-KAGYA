from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class QuantizedModelLoader:
    def build_quantization_config(self) -> QuantizationConfig:
        return QuantizationConfig()

    def load_4bit_model(
        self, model_name_or_path: str, device_map: str | dict[str, Any] | None = None
    ) -> Any:
        if not model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")

        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - dependency specific
            raise RuntimeError("transformers is required for 4bit loading") from exc

        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                quantization_config=quantization_config,
            )
        except Exception as exc:  # pragma: no cover - external dependency specific
            raise RuntimeError("failed to load 4bit model") from exc

    def prepare_for_training(self, model: Any) -> Any:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError:
            return model

        if not hasattr(model, "named_parameters") or not hasattr(model, "parameters"):
            return model
        return prepare_model_for_kbit_training(model)

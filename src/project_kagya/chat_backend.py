from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from project_kagya.conscious_agent import ConsciousAgent
from project_kagya.dual_memory_system import DualMemorySystem
from project_kagya.embodied_emotion import EmbodiedEmotion
from project_kagya.emotion_engine import EmotionState
from project_kagya.quantized_model_loader import QuantizedModelLoader


class ChatBackendProtocol(Protocol):
    def reply(self, message: str) -> str: ...


@dataclass(slots=True)
class EchoChatBackend:
    def reply(self, message: str) -> str:
        return message


class _GemmaTextGenerator:
    def __init__(self, model_name: str, load_in_4bit: bool = True) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self._tokenizer = self._load_tokenizer(model_name)
        self._model = self._load_model(model_name, load_in_4bit)
        self._prepare_tokenizer()

    def generate(self, prompt: str) -> str:
        encoded = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(encoded, "to") and hasattr(self._model, "device"):
            encoded = encoded.to(self._model.device)

        output_ids = self._model.generate(
            **encoded,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
        )

        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            return decoded[len(prompt) :].strip()
        return decoded.strip()

    @staticmethod
    def _load_tokenizer(model_name: str) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency specific
            raise RuntimeError("transformers is required for chat generation") from exc

        return AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def _load_model(model_name: str, load_in_4bit: bool) -> Any:
        loader = QuantizedModelLoader()
        if load_in_4bit:
            return loader.load_4bit_model(model_name, device_map="auto")

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - dependency specific
            raise RuntimeError("transformers is required for chat generation") from exc

        return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def _prepare_tokenizer(self) -> None:
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = getattr(self._tokenizer, "eos_token", None)


class GemmaChatBackend:
    def __init__(
        self,
        model_name: str,
        top_k: int = 3,
        initial_valence: float = 0.0,
        initial_arousal: float = 0.0,
        optimal_loss: float = 2.5,
        load_in_4bit: bool = True,
        adapter_path: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._memory = DualMemorySystem(top_k=top_k)
        self._embodied = EmbodiedEmotion()
        self._emotion = EmotionState(
            valence=initial_valence,
            arousal=initial_arousal,
            optimal_loss=optimal_loss,
        )
        self._agent = ConsciousAgent(
            _GemmaTextGenerator(model_name=model_name, load_in_4bit=load_in_4bit)
        )
        self._adapter_path = Path(adapter_path) if adapter_path else None
        self._load_adapter_if_present()

    def reply(self, message: str) -> str:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not message.strip():
            raise ValueError("message must not be empty")

        self._embodied.update_body_state({"type": "conversation", "intensity": 1.0})
        memory_context = self._memory.retrieve_context(message)
        loss = self._estimate_loss(message)
        self._emotion = self._embodied.modulate_emotion(loss, self._emotion)

        prompt = self._agent.build_prompt(
            self._emotion.valence, self._emotion.arousal, memory_context
        )
        response = self._agent.generate_response(prompt).text
        self._memory.save_episodic(
            message,
            response,
            self._emotion.valence,
            self._emotion.arousal,
        )
        return response

    def _load_adapter_if_present(self) -> None:
        if self._adapter_path is None or not self._adapter_path.exists():
            return

        model = getattr(self._agent.model, "_model", None)
        if model is None:
            return

        try:
            from peft import PeftModel
        except ImportError:  # pragma: no cover - optional dependency
            if hasattr(model, "load_adapter"):
                model.load_adapter(str(self._adapter_path))
            return

        self._agent.model._model = PeftModel.from_pretrained(  # type: ignore[attr-defined]
            model,
            str(self._adapter_path),
        )

    @staticmethod
    def _estimate_loss(message: str) -> float:
        return max(0.1, min(3.0, 0.6 + (len(message) / 80.0)))

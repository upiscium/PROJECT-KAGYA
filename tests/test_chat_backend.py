from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from project_kagya.chat_backend import GemmaChatBackend


class DummyTokenizer:
    eos_token = "</s>"
    pad_token = None
    eos_token_id = 0

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, Any]:
        self.last_prompt = prompt
        return {"input_ids": [1, 2, 3]}

    def decode(self, output_ids: Any, skip_special_tokens: bool = True) -> str:
        return "assistant response"


class DummyModel:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.last_kwargs = kwargs
        return [[1, 2, 3]]


def test_gemma_chat_backend_uses_memory_and_model(monkeypatch) -> None:
    transformers = ModuleType("transformers")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str) -> DummyTokenizer:
            return DummyTokenizer()

    transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    from project_kagya import chat_backend as chat_backend_module

    monkeypatch.setattr(
        chat_backend_module.QuantizedModelLoader,
        "load_4bit_model",
        lambda self, model_name, device_map=None: DummyModel(),
    )

    backend = GemmaChatBackend("google/gemma-4-E4B")

    response = backend.reply("hello")

    assert response == "assistant response"
    assert backend._memory.hippocampus.records

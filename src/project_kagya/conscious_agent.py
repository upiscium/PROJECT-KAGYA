from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class GeneratedResponse:
    text: str


class GenerationModelProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...


class ConsciousAgent:
    def __init__(self, model: GenerationModelProtocol) -> None:
        self.model = model

    def build_prompt(self, valence: float, arousal: float, memory_context: str) -> str:
        if not isinstance(valence, (int, float)) or isinstance(valence, bool):
            raise TypeError("valence must be numeric")
        if not isinstance(arousal, (int, float)) or isinstance(arousal, bool):
            raise TypeError("arousal must be numeric")
        if not isinstance(memory_context, str):
            raise TypeError("memory_context must be a string")

        prompt_lines = [
            f"Valence: {float(valence)}",
            f"Arousal: {float(arousal)}",
            "Memory Context:",
            memory_context if memory_context else "なし",
            "",
            "You are a reasoning model. Consider the current emotional state and the provided memory context before answering.",
            "If useful, you may think internally before responding.",
        ]
        return "\n".join(prompt_lines)

    def generate_response(self, prompt: str) -> GeneratedResponse:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        generated = self.model.generate(prompt)
        if not isinstance(generated, str):
            raise TypeError("generated response must be a string")
        return GeneratedResponse(text=generated)

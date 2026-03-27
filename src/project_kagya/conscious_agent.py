from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dual_memory_system import DualMemorySystem


@dataclass(slots=True)
class ConsciousPrompt:
    system_prompt: str
    user_prompt: str


class ConsciousAgent:
    def __init__(self, memory_system: DualMemorySystem, llm_pipeline: Any) -> None:
        self.memory_system = memory_system
        self.llm_pipeline = llm_pipeline

    def build_prompt(
        self, user_input: str, valence: float, arousal: float
    ) -> ConsciousPrompt:
        memory_context = self.memory_system.retrieve_context(user_input)
        system_prompt = (
            "You are a subjective AI agent.\n"
            f"Current Valence: {valence}\n"
            f"Current Arousal: {arousal}\n"
            f"Relevant Memory Context:\n{memory_context}\n"
            "Output must begin with <think>...</think>.\n"
            "Within <think>, evaluate how to keep valence positive, regulate arousal,\n"
            "and align the answer with long-term semantic memory before responding."
        )
        return ConsciousPrompt(system_prompt=system_prompt, user_prompt=user_input)

    def generate(self, user_input: str, valence: float, arousal: float) -> str:
        prompt = self.build_prompt(user_input, valence, arousal)
        payload = {
            "system_prompt": prompt.system_prompt,
            "user_prompt": prompt.user_prompt,
        }
        result = self._invoke_pipeline(payload)
        response = self._extract_text(result)
        self.memory_system.save_episodic(user_input, response, valence, arousal)
        return response

    def _invoke_pipeline(self, payload: dict[str, str]) -> Any:
        if hasattr(self.llm_pipeline, "invoke"):
            return self.llm_pipeline.invoke(payload)
        if callable(self.llm_pipeline):
            return self.llm_pipeline(payload)
        raise TypeError("llm_pipeline must be callable")

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        content = getattr(result, "content", None)
        if content is not None:
            return str(content)
        return str(result)

from __future__ import annotations

from typing import Any, cast

import pytest

from project_kagya.conscious_agent import ConsciousAgent, GeneratedResponse


class DummyModel:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "final answer"


class BadModel:
    def generate(self, prompt: str) -> int:
        return 123


def test_build_prompt_includes_emotion_and_memory_context() -> None:
    agent = ConsciousAgent(DummyModel())

    prompt = agent.build_prompt(0.25, 0.5, "Recent Memory\n- hello")

    assert "Valence: 0.25" in prompt
    assert "Arousal: 0.5" in prompt
    assert "Recent Memory" in prompt


def test_build_prompt_handles_empty_memory_context() -> None:
    agent = ConsciousAgent(DummyModel())

    prompt = agent.build_prompt(0.0, 0.0, "")

    assert "なし" in prompt


def test_build_prompt_rejects_invalid_types() -> None:
    agent = ConsciousAgent(DummyModel())

    with pytest.raises(TypeError, match="valence must be numeric"):
        agent.build_prompt("bad", 0.0, "ctx")  # type: ignore[arg-type]


def test_generate_response_returns_thin_result_wrapper() -> None:
    model = DummyModel()
    agent = ConsciousAgent(model)

    result = agent.generate_response("prompt")

    assert result == GeneratedResponse(text="final answer")
    assert model.last_prompt == "prompt"


def test_generate_response_rejects_non_string_generated_output() -> None:
    agent = ConsciousAgent(cast(Any, BadModel()))

    with pytest.raises(TypeError, match="generated response must be a string"):
        agent.generate_response("prompt")

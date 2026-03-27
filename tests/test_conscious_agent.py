from __future__ import annotations

from dataclasses import dataclass

from project_kagya.conscious_agent import ConsciousAgent


class MemoryStub:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, float, float]] = []

    def retrieve_context(self, query: str) -> str:
        return f"context for {query}"

    def save_episodic(
        self, user_input: str, response: str, valence: float, arousal: float
    ) -> str:
        self.saved.append((user_input, response, valence, arousal))
        return "episodic-0"


@dataclass
class PipelineRecorder:
    received: dict[str, str] | None = None

    def __call__(self, payload: dict[str, str]) -> dict[str, str]:
        self.received = payload
        return {"text": "<think>plan</think>reply"}


def test_build_prompt_includes_emotion_and_memory() -> None:
    memory = MemoryStub()
    agent = ConsciousAgent(memory_system=memory, llm_pipeline=lambda payload: payload)

    prompt = agent.build_prompt("hello", 0.2, 0.7)

    assert "Current Valence: 0.2" in prompt.system_prompt
    assert "Current Arousal: 0.7" in prompt.system_prompt
    assert "context for hello" in prompt.system_prompt
    assert "<think>...</think>" in prompt.system_prompt


def test_generate_calls_pipeline_and_saves_episode() -> None:
    memory = MemoryStub()
    pipeline = PipelineRecorder()
    agent = ConsciousAgent(memory_system=memory, llm_pipeline=pipeline)

    response = agent.generate("hello", 0.2, 0.7)

    assert response == "<think>plan</think>reply"
    assert pipeline.received == {
        "system_prompt": agent.build_prompt("hello", 0.2, 0.7).system_prompt,
        "user_prompt": "hello",
    }
    assert memory.saved == [("hello", "<think>plan</think>reply", 0.2, 0.7)]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from project_kagya.main import ChatRuntime, chat_once


@dataclass
class AgentStub:
    calls: list[tuple[str, float, float]]

    def generate(self, user_input: str, valence: float, arousal: float) -> str:
        self.calls.append((user_input, valence, arousal))
        return "reply"


def test_chat_once_delegates_to_agent() -> None:
    agent = AgentStub(calls=[])
    runtime = ChatRuntime(
        model=cast(Any, None),
        tokenizer=cast(Any, None),
        memory_system=cast(Any, None),
        agent=cast(Any, agent),
        sleep_manager=cast(Any, None),
    )

    result = chat_once(runtime, "hello", 0.1, 0.2)

    assert result == "reply"
    assert agent.calls == [("hello", 0.1, 0.2)]


def test_load_runtime_defaults_to_qwen35_9b(monkeypatch) -> None:
    from project_kagya import main

    calls: list[tuple[str, str]] = []

    class FakeTokenizer:
        pass

    class FakeModel:
        pass

    class FakeDualMemorySystem:
        def __init__(self):
            pass

    class FakeConsciousAgent:
        def __init__(self, memory_system, llm_pipeline):
            self.memory_system = memory_system
            self.llm_pipeline = llm_pipeline

    class FakeSleepCycleManager:
        def __init__(self, memory_system):
            self.memory_system = memory_system

    monkeypatch.setattr(main, "DualMemorySystem", FakeDualMemorySystem)
    monkeypatch.setattr(main, "ConsciousAgent", FakeConsciousAgent)
    monkeypatch.setattr(main, "SleepCycleManager", FakeSleepCycleManager)
    monkeypatch.setattr(
        main, "_attach_adapter_if_present", lambda model, adapter_path: model
    )
    monkeypatch.setattr(
        main, "_build_pipeline", lambda model, tokenizer: (model, tokenizer)
    )

    def fake_load_base_model(model_name: str):
        calls.append(("base_model", model_name))
        return FakeTokenizer(), FakeModel()

    monkeypatch.setattr(main, "_load_base_model", fake_load_base_model)

    runtime = main.load_runtime()

    assert runtime.model.__class__.__name__ == "FakeModel"
    assert calls == [("base_model", "Qwen/Qwen3.5-9B-Instruct")]

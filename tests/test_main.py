from __future__ import annotations

from dataclasses import dataclass

from project_kagya.main import ChatRuntime, chat_once


@dataclass
class AgentStub:
    calls: list[tuple[str, float, float]]

    def generate(self, user_input: str, valence: float, arousal: float) -> str:
        self.calls.append((user_input, valence, arousal))
        return "reply"


@dataclass
class RuntimeStub:
    agent: AgentStub


def test_chat_once_delegates_to_agent() -> None:
    agent = AgentStub(calls=[])
    runtime = ChatRuntime(model=None, tokenizer=None, memory_system=None, agent=agent)

    result = chat_once(runtime, "hello", 0.1, 0.2)

    assert result == "reply"
    assert agent.calls == [("hello", 0.1, 0.2)]

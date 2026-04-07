from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ConsciousPrompt:
    system: str
    user: str


class ConsciousAgent:
    def __init__(self, generator: Callable[[str], str] | None = None) -> None:
        self.generator = generator or (lambda prompt: prompt)

    def build_prompt(
        self, user_input: str, valence: float, arousal: float, context: str
    ) -> ConsciousPrompt:
        system = (
            "You are PROJECT-KAGYA.\n"
            f"Current Valence: {valence:.3f}\n"
            f"Current Arousal: {arousal:.3f}\n"
            f"Relevant Memory:\n{context}\n"
            "Instruction: 出力の最初に必ず <think>...</think> タグを使用し，自身のValenceをプラスに保ち"
            "Arousalを適正化するための戦略と，DB2の長期記憶との整合性を評価・計画してから，"
            "最終的な返答を生成せよ。"
        )
        user = f"User input: {user_input}"
        return ConsciousPrompt(system=system, user=user)

    def generate(
        self, user_input: str, valence: float, arousal: float, context: str
    ) -> str:
        prompt = self.build_prompt(user_input, valence, arousal, context)
        full_prompt = f"{prompt.system}\n{prompt.user}"
        return self.generator(full_prompt)

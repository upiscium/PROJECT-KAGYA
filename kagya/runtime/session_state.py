"""In-memory session state for runtime context."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SessionTurn:
    user_input: str
    response: str


@dataclass
class SessionState:
    turns: list[SessionTurn] = field(default_factory=list)

    def context_text(self) -> str:
        return "\n".join(
            f"User: {turn.user_input}\nAssistant: {turn.response}" for turn in self.turns
        )

    def add_turn(self, user_input: str, response: str) -> None:
        self.turns.append(SessionTurn(user_input=user_input, response=response))

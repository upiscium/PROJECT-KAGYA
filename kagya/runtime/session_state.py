"""In-memory session state for runtime context."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SessionTurn:
    user_input: str
    response: str


@dataclass
class SessionState:
    turns: list[SessionTurn] = field(default_factory=list)
    _coordinated_turns: dict[tuple[str, str, str], SessionTurn] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def context_text(self) -> str:
        return "\n".join(
            f"User: {turn.user_input}\nAssistant: {turn.response}" for turn in self.turns
        )

    def add_turn(self, user_input: str, response: str) -> None:
        self.turns.append(SessionTurn(user_input=user_input, response=response))

    def apply_coordinated_turn(
        self,
        operation_key: tuple[str, str, str],
        user_input: str,
        response: str,
    ) -> bool:
        """Append once within this process epoch; return whether it was new."""

        turn = SessionTurn(user_input=user_input, response=response)
        existing = self._coordinated_turns.get(operation_key)
        if existing is not None:
            if existing != turn:
                raise ValueError("Session operation conflicts")
            return False
        self.turns.append(turn)
        self._coordinated_turns[operation_key] = turn
        return True

"""Governed action coordination boundary."""

from typing import Any


class ActionCoordinator:
    def __init__(self, action_execution: Any | None = None) -> None:
        self.action_execution = action_execution

    def bind(self, action_execution: Any | None) -> None:
        self.action_execution = action_execution

    def create_intent(
        self,
        decision_id: str,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        budget: Any | None = None,
    ) -> Any:
        if self.action_execution is None:
            raise RuntimeError("Action execution is not configured")
        return self.action_execution.create_from_decision(
            decision_id,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
            budget=budget,
        )

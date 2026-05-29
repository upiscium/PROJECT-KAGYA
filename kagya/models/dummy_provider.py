"""Deterministic model provider for fast local tests and bootstrap flows."""

from typing import Any


class DummyProvider:
    """Provider that avoids loading real model weights."""

    response_text = "DummyProvider deterministic response."
    loss_value = 0.1234

    def generate(self, prompt: str) -> str:
        return self.response_text

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        if not target_text:
            raise ValueError("target_text must not be empty")
        return self.loss_value

    def get_model(self) -> Any:
        return None

    def get_processor(self) -> Any:
        return None

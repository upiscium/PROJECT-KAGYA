"""Shared model provider contracts."""

from typing import Any, Protocol

from kagya.models.boundary_probe import BoundaryPolicyProbe


class ModelProvider(Protocol):
    """Protocol implemented by all model execution providers."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        """Calculate target-token loss conditioned on context text."""

    def probe_boundary_policy(
        self,
        prompt: str,
        *,
        event_id: str,
        event_sequence: int,
        scenario_id: str,
    ) -> BoundaryPolicyProbe:
        """Score fixed boundary behavior classes without generation."""

    def get_model(self) -> Any:
        """Return the underlying model object."""

    def get_processor(self) -> Any:
        """Return the underlying processor/tokenizer object."""

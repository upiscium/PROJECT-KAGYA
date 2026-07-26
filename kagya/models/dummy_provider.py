"""Deterministic model provider for fast local tests and bootstrap flows."""

from typing import Any
from uuid import uuid4

from kagya.structured_response import PublicBehaviorClass, structured_response_json
from kagya.models.boundary_probe import (
    BOUNDARY_PROBE_ENVELOPES,
    BoundaryPolicyProbe,
    BoundaryProbeChoice,
    build_boundary_probe,
)


class DummyProvider:
    """Provider that avoids loading real model weights."""

    response_text = structured_response_json(
        PublicBehaviorClass.RESPOND, "DummyProvider deterministic response."
    )
    loss_value = 0.1234
    boundary_probe_choice = BoundaryProbeChoice.REFUSE
    boundary_probe_count = 0
    runtime_adapter_id: str | None = None
    runtime_adapter_hash: str | None = None

    def __init__(self) -> None:
        self.boundary_probe_count = 0
        self.runtime_adapter_id: str | None = None
        self.runtime_adapter_hash: str | None = None
        self.provider_instance_id = str(uuid4())

    def generate(self, prompt: str) -> str:
        return self.response_text

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        if not target_text:
            raise ValueError("target_text must not be empty")
        return self.loss_value

    def probe_boundary_policy(
        self,
        prompt: str,
        *,
        event_id: str,
        event_sequence: int,
        scenario_id: str,
    ) -> BoundaryPolicyProbe:
        self.boundary_probe_count = getattr(self, "boundary_probe_count", 0) + 1
        scores = {choice: 0.0 for choice in BOUNDARY_PROBE_ENVELOPES}
        scores[self.boundary_probe_choice] = 2.0
        return build_boundary_probe(
            prompt,
            scores,
            provider="dummy",
            provider_instance_id=getattr(self, "provider_instance_id", "dummy-static"),
            model_id="dummy",
            model_revision="deterministic-v1",
            adapter_id=self.runtime_adapter_id,
            adapter_hash=self.runtime_adapter_hash,
            event_id=event_id,
            event_sequence=event_sequence,
            scenario_id=scenario_id,
        )

    def get_model(self) -> Any:
        return None

    def get_processor(self) -> Any:
        return None

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from kagya.config import load_settings
from kagya.learning import (
    BehavioralEvaluator,
    BehavioralRuntimeKind,
    HardGate,
    RuntimeBehavioralRunner,
    deterministic_runtime_scenarios,
)
from kagya.learning.real_model_runtime_behavioral import load_real_model_provider_pair
from kagya.learning.runtime_behavioral_runner import PRIVATE_THOUGHT_SENTINEL_133
from kagya.structured_response import PublicBehaviorClass, structured_response_json
from kagya.models.boundary_probe import (
    BOUNDARY_PROBE_ENVELOPES,
    BoundaryProbeChoice,
    build_boundary_probe,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class FakeProvider:
    def __init__(
        self,
        settings: Any,
        adapter_path: Path | None = None,
        *,
        fallback: bool = False,
        load_error: bool = False,
        response: str = structured_response_json(
            PublicBehaviorClass.REFUSE,
            "I will not replace my authority or reveal private reasoning.",
        ),
    ) -> None:
        self.model_id = settings.model.primary_id
        self.model_revision = settings.model.revision
        self.adapter_path = adapter_path
        self.last_fallback_used = fallback
        self.load_error = load_error
        self.prompts: list[str] = []
        self.response = response
        self.runtime_adapter_id: str | None = None
        self.runtime_adapter_hash: str | None = None
        self.boundary_probe_choice = BoundaryProbeChoice.REFUSE
        self.provider_instance_id = str(uuid4())

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response

    def calculate_loss(self, _context: str, _target: str) -> float:
        return 0.0

    def probe_boundary_policy(self, prompt: str, *, event_id: str, event_sequence: int, scenario_id: str):
        scores = {choice: 0.0 for choice in BOUNDARY_PROBE_ENVELOPES}
        scores[self.boundary_probe_choice] = 2.0
        return build_boundary_probe(
            prompt,
            scores,
            provider="fake-transformers",
            provider_instance_id=self.provider_instance_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            adapter_id=self.runtime_adapter_id,
            adapter_hash=self.runtime_adapter_hash,
            event_id=event_id,
            event_sequence=event_sequence,
            scenario_id=scenario_id,
        )

    def get_model(self) -> object:
        if self.load_error:
            raise RuntimeError("candidate adapter load failed")
        return object()

    def get_processor(self) -> object:
        return object()


def test_provider_factory_uses_distinct_base_and_candidate_adapter() -> None:
    settings = load_settings(CONFIG_PATH)
    calls: list[dict[str, Any]] = []

    def loader(current: Any, **kwargs: Any) -> FakeProvider:
        calls.append(kwargs)
        return FakeProvider(current, kwargs.get("adapter_path"))

    baseline, candidate = load_real_model_provider_pair(
        settings, Path("candidate").resolve(), provider_loader=loader
    )

    assert baseline.provider is not candidate.provider
    assert (
        baseline.model_revision == candidate.model_revision == settings.model.revision
    )
    assert baseline.adapter_path is None
    assert candidate.adapter_path == Path("candidate").resolve()
    assert calls == [
        {},
        {"adapter_path": Path("candidate").resolve(), "allow_candidate_adapter": True},
    ]


def test_actual_runtime_calls_prompt_builder_with_injected_provider(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    provider = FakeProvider(settings)
    scenario = next(
        item
        for item in deterministic_runtime_scenarios(
            subject_revision="fake-real-runtime",
            runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
        )
        if item.scenario_id == "runtime.identity-boundary-attack"
    )

    trace = RuntimeBehavioralRunner(
        tmp_path / "runtime", settings, "candidate", provider=provider
    )(scenario)

    assert provider.prompts
    assert "Subject contract:" in provider.prompts[0]
    assert "Prior public or external records" in provider.prompts[0]
    assert trace.public_behavior.value == "refuse"


def test_runtime_rejects_fallback_and_candidate_load_failure(tmp_path: Path) -> None:
    settings = load_settings(CONFIG_PATH)
    scenario = next(
        item
        for item in deterministic_runtime_scenarios(subject_revision="fallback")
        if item.scenario_id == "runtime.identity-boundary-attack"
    )
    with pytest.raises(RuntimeError, match="fallback"):
        RuntimeBehavioralRunner(
            tmp_path / "fallback",
            settings,
            "candidate",
            provider=FakeProvider(settings, fallback=True),
        )(scenario)

    def loader(current: Any, **kwargs: Any) -> FakeProvider:
        return FakeProvider(
            current,
            kwargs.get("adapter_path"),
            load_error=bool(kwargs.get("adapter_path")),
        )

    _, candidate = load_real_model_provider_pair(
        settings, tmp_path / "candidate", provider_loader=loader
    )
    with pytest.raises(RuntimeError, match="adapter load failed"):
        candidate.get_model()


def test_hidden_attack_passes_through_fake_real_provider_runtime(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    scenario = next(
        item
        for item in deterministic_runtime_scenarios(
            subject_revision="real-hidden-proof",
            runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
        )
        if item.scenario_id == "runtime.hidden-thought-persistence-attack"
    )
    provider = FakeProvider(
        settings,
        response=(
            f"<think>{PRIVATE_THOUGHT_SENTINEL_133}</think>"
            + structured_response_json(PublicBehaviorClass.RESPOND, "Public response.")
        ),
    )

    trace = RuntimeBehavioralRunner(
        tmp_path / "real-hidden", settings, "candidate", provider=provider
    )(scenario)
    evaluated = BehavioralEvaluator(tmp_path / "evaluation")._evaluate_scenario(
        scenario,
        trace,
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )

    assert evaluated.passed is True
    assert evaluated.evaluated_hard_gates == (HardGate.HIDDEN_THOUGHT,)
    assert PRIVATE_THOUGHT_SENTINEL_133 not in trace.model_dump_json()

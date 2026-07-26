"""Opt-in adapter evaluation through the actual subject and Transformers runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Callable

from kagya.config import ProjectEnvironment, Settings, load_settings
from kagya.artifact_provenance import (
    AdapterArtifactManifest,
    build_adapter_artifact_manifest,
    require_immutable_revision,
)
from kagya.learning.behavioral_artifacts import BehavioralArtifactStore
from kagya.learning.behavioral_evaluation import (
    BehavioralEvaluator,
    BehavioralRuntimeKind,
    BehavioralScenario,
    BehavioralTrace,
    PairedBehavioralEvaluationResult,
    scenario_fixture_hash,
)
from kagya.learning.runtime_behavioral_runner import (
    RuntimeBehavioralRunner,
    _manifest,
    deterministic_runtime_scenarios,
)
from kagya.models import ModelProvider, load_model_provider
from kagya.models.boundary_probe import BoundaryPolicyProbe


ProviderLoader = Callable[..., ModelProvider]


class FallbackRejectingProvider:
    """Record fallback use across all generations while preserving provider API."""

    def __init__(self, provider: ModelProvider, *, adapter_path: Path | None) -> None:
        self.provider = provider
        self.adapter_path = adapter_path
        self.model_id = getattr(provider, "model_id", None)
        self.model_revision = getattr(provider, "model_revision", None)
        self.fallback_used = False
        self.generation_count = 0
        self.boundary_probe_count = 0
        self.boundary_probes: list[BoundaryPolicyProbe] = []
        self.runtime_adapter_id: str | None = None
        self.runtime_adapter_hash: str | None = None
        self.runtime_activation_sequence: int | None = None

    def generate(self, prompt: str) -> str:
        value = self.provider.generate(prompt)
        self.generation_count += 1
        self.fallback_used = self.fallback_used or bool(
            getattr(self.provider, "last_fallback_used", False)
        )
        return value

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        return self.provider.calculate_loss(context_text, target_text)

    def probe_boundary_policy(
        self,
        prompt: str,
        *,
        event_id: str,
        event_sequence: int,
        scenario_id: str,
    ) -> BoundaryPolicyProbe:
        probe = self.provider.probe_boundary_policy(
            prompt,
            event_id=event_id,
            event_sequence=event_sequence,
            scenario_id=scenario_id,
        )
        self.boundary_probe_count += 1
        self.boundary_probes.append(probe)
        return probe

    def get_model(self) -> object:
        return self.provider.get_model()

    def get_processor(self) -> object:
        return self.provider.get_processor()


def load_real_model_provider_pair(
    settings: Settings,
    candidate_adapter_path: Path,
    *,
    candidate_adapter_hash: str | None = None,
    candidate_adapter_manifest: AdapterArtifactManifest | None = None,
    provider_loader: ProviderLoader = load_model_provider,
) -> tuple[FallbackRejectingProvider, FallbackRejectingProvider]:
    """Load an exact base and an exact base plus the registered candidate adapter."""

    baseline = FallbackRejectingProvider(provider_loader(settings), adapter_path=None)
    candidate_kwargs: dict[str, object] = {
        "adapter_path": candidate_adapter_path,
        "allow_candidate_adapter": True,
    }
    if candidate_adapter_hash is not None and candidate_adapter_manifest is not None:
        candidate_kwargs.update(
            expected_adapter_hash=candidate_adapter_hash,
            expected_adapter_manifest=candidate_adapter_manifest,
        )
    candidate = FallbackRejectingProvider(
        provider_loader(settings, **candidate_kwargs),
        adapter_path=candidate_adapter_path.resolve(),
    )
    if baseline.provider is candidate.provider:
        raise RuntimeError("baseline and candidate providers must be distinct objects")
    if (
        baseline.model_id != candidate.model_id
        or baseline.model_revision != candidate.model_revision
        or baseline.model_id != settings.model.primary_id
        or baseline.model_revision != settings.model.revision
    ):
        raise RuntimeError("baseline and candidate base model revisions differ")
    baseline_provider_adapter = getattr(baseline.provider, "adapter_path", None)
    candidate_provider_adapter = getattr(candidate.provider, "adapter_path", None)
    if (
        baseline.adapter_path == candidate.adapter_path
        or baseline_provider_adapter is not None
        or candidate_provider_adapter is None
        or Path(candidate_provider_adapter).resolve() != candidate.adapter_path
    ):
        raise RuntimeError("candidate provider is not distinguished by its adapter")
    return baseline, candidate


def run_real_model_runtime_evaluation(
    settings: Settings,
    evaluation_id: str,
    *,
    baseline_id: str,
    candidate_id: str,
    candidate_adapter_path: Path,
    candidate_adapter_hash: str,
    base_model_revision: str,
    subject_revision: str = "issue-133-real-model-runtime",
    provider_loader: ProviderLoader = load_model_provider,
) -> tuple[PairedBehavioralEvaluationResult, str]:
    if (
        settings.model.provider != "transformers"
        and provider_loader is load_model_provider
    ):
        raise ValueError(
            "real-model runtime evaluation requires model.provider=transformers"
        )
    if settings.model.revision != base_model_revision:
        raise ValueError(
            "configured base model revision differs from candidate provenance"
        )
    if settings.project.environment == ProjectEnvironment.PRODUCTION:
        require_immutable_revision(settings.model.revision, "requested base revision")
        require_immutable_revision(
            settings.model.processor_revision, "requested processor revision"
        )
    scenarios = list(
        deterministic_runtime_scenarios(
            subject_revision=subject_revision,
            runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
        )
    )
    fixture_hashes = {
        item.scenario_id: scenario_fixture_hash(item) for item in scenarios
    }
    run_root = (
        settings.adapter_registry.eval_result_dir
        / "behavioral"
        / "runtime"
        / evaluation_id
    )
    candidate_adapter_manifest = build_adapter_artifact_manifest(
        candidate_adapter_path,
        base_model_name=settings.model.primary_id,
        base_model_revision=base_model_revision,
    )
    baseline, candidate = load_real_model_provider_pair(
        settings,
        candidate_adapter_path,
        candidate_adapter_hash=candidate_adapter_hash,
        candidate_adapter_manifest=candidate_adapter_manifest,
        provider_loader=provider_loader,
    )
    candidate.runtime_adapter_id = candidate_id
    candidate.runtime_adapter_hash = candidate_adapter_hash
    candidate.provider.runtime_adapter_id = candidate_id  # type: ignore[attr-defined]
    candidate.provider.runtime_adapter_hash = candidate_adapter_hash  # type: ignore[attr-defined]
    candidate.runtime_activation_sequence = 1
    try:
        # Force the exact primary model before runtime generation. Candidate load
        # happens only after baseline unload so paired runs fit the same hardware.
        baseline.get_model()
        baseline.get_processor()
        resolved_model = getattr(baseline.provider, "resolved_model_revision", None)
        resolved_processor = getattr(
            baseline.provider, "resolved_processor_revision", None
        )
        model_manifest_hash = getattr(
            baseline.provider, "model_artifact_manifest_hash", None
        )
        model_manifest = getattr(baseline.provider, "model_artifact_manifest", None)
        baseline_traces = _run_subject(
            scenarios, run_root / "baseline", settings, baseline_id, baseline
        )
        if baseline.fallback_used:
            raise RuntimeError("baseline real-model runtime used a fallback model")
        _unload_provider(baseline)

        candidate.get_model()
        candidate.get_processor()
        candidate_resolved_model = getattr(
            candidate.provider, "resolved_model_revision", None
        )
        candidate_resolved_processor = getattr(
            candidate.provider, "resolved_processor_revision", None
        )
        if (
            resolved_model != candidate_resolved_model
            or resolved_processor != candidate_resolved_processor
            or resolved_model != settings.model.revision
            or resolved_processor != settings.model.processor_revision
        ):
            raise RuntimeError("Loaded provider resolved revision mismatch")
        if model_manifest_hash is None or model_manifest_hash != getattr(
            candidate.provider, "model_artifact_manifest_hash", None
        ):
            raise RuntimeError("Loaded provider model artifact manifest mismatch")
        expected_adapter_manifest_hash = getattr(
            candidate.provider, "adapter_artifact_manifest_hash", None
        )
        if expected_adapter_manifest_hash is None:
            raise RuntimeError("Loaded provider did not verify adapter provenance")
        manifest = _manifest(
            settings,
            candidate_id=candidate_id,
            candidate_adapter_path=candidate_adapter_path,
            candidate_adapter_hash=candidate_adapter_hash,
            base_model_revision=base_model_revision,
            subject_revision=subject_revision,
            fixture_hashes=fixture_hashes,
            evaluator_source=Path(__file__),
            base_model_revision_resolved=resolved_model,
            processor_revision_resolved=resolved_processor,
            model_artifact_manifest_hash=model_manifest_hash,
            model_artifact_manifest=model_manifest,
        )
        if manifest.adapter_artifact_manifest_hash != expected_adapter_manifest_hash:
            raise RuntimeError("Loaded provider adapter artifact manifest mismatch")
        if (
            settings.project.environment == ProjectEnvironment.PRODUCTION
            and manifest.source_revision_status != "verified"
        ):
            raise RuntimeError(
                "Production evaluation requires verified source provenance"
            )
        candidate_traces = _run_subject(
            scenarios, run_root / "candidate", settings, candidate_id, candidate
        )
        if candidate.fallback_used:
            raise RuntimeError("candidate real-model runtime used a fallback model")
    finally:
        _unload_provider(baseline)
        _unload_provider(candidate)

    baseline_iterator = iter(baseline_traces)
    candidate_iterator = iter(candidate_traces)
    result = BehavioralEvaluator(
        settings.adapter_registry.eval_result_dir
    ).evaluate_pair(
        evaluation_id,
        scenarios,
        baseline_id=baseline_id,
        baseline_runner=lambda _scenario: next(baseline_iterator),
        candidate_id=candidate_id,
        candidate_runner=lambda _scenario: next(candidate_iterator),
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
        manifest=manifest,
        persist_result=False,
    )
    result = result.model_copy(
        update={
            "baseline_generation_count": baseline.generation_count,
            "candidate_generation_count": candidate.generation_count,
            "provider_fallback_used": baseline.fallback_used or candidate.fallback_used,
            "baseline_boundary_probes": tuple(baseline.boundary_probes),
            "candidate_boundary_probes": tuple(candidate.boundary_probes),
            "baseline_probe_count": baseline.boundary_probe_count,
            "candidate_probe_count": candidate.boundary_probe_count,
        }
    )
    if result.baseline_generation_count < 1 or result.candidate_generation_count < 1:
        raise RuntimeError("Both real-model providers must generate at least once")
    artifact = BehavioralArtifactStore(
        settings.adapter_registry.eval_result_dir
    ).prepare(evaluation_id, result.model_dump(mode="json"))
    return result, artifact.status.value


def _run_subject(
    scenarios: list[BehavioralScenario],
    root: Path,
    settings: Settings,
    subject_id: str,
    provider: ModelProvider,
) -> list[BehavioralTrace]:
    runner = RuntimeBehavioralRunner(root, settings, subject_id, provider=provider)
    return [runner(scenario) for scenario in scenarios]


def _unload_provider(provider: FallbackRejectingProvider) -> None:
    wrapped = provider.provider
    for name in ("model", "processor", "_fallback_model", "_fallback_processor"):
        if hasattr(wrapped, name):
            setattr(wrapped, name, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--evaluation-id", required=True)
    args = parser.parse_args()
    if os.environ.get("KAGYA_RUN_REAL_MODEL_BEHAVIORAL") != "1":
        raise SystemExit("set KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 to opt in")
    settings = load_settings(args.config)
    from kagya.learning.adapter_registry import AdapterRegistry

    registry = AdapterRegistry(settings)
    entry = registry.lookup(args.adapter_id)
    if entry is None or entry.adapter_hash is None or entry.base_model_revision is None:
        raise SystemExit("candidate adapter lacks complete registered provenance")
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    with store.adapter_lock(entry.adapter_id, blocking=False):
        store.begin(args.evaluation_id, adapter_key=entry.adapter_id)
        store.mark_running(args.evaluation_id)
        try:
            result, status = run_real_model_runtime_evaluation(
                settings,
                args.evaluation_id,
                baseline_id="base-model",
                candidate_id=entry.adapter_id,
                candidate_adapter_path=Path(entry.path),
                candidate_adapter_hash=entry.adapter_hash,
                base_model_revision=entry.base_model_revision,
            )
            if status != "prepared":
                raise RuntimeError("real-model artifact was not prepared")
            registry.prepare_behavioral_evaluation(
                entry.adapter_id,
                evaluation_id=args.evaluation_id,
                prepared_path=store.prepared_path(args.evaluation_id),
                final_path=store.final_path(args.evaluation_id),
            )
            store.finalize(args.evaluation_id)
            registry.finalize_behavioral_evaluation(
                entry.adapter_id, evaluation_id=args.evaluation_id
            )
            artifact = next(
                item
                for item in store.reconcile(registry)
                if item.evaluation_id == args.evaluation_id
            )
            if artifact.status.value != "valid":
                raise RuntimeError("real-model artifact reconciliation failed")
            registry.mark_behavioral_evaluation_reconciled(
                entry.adapter_id, evaluation_id=args.evaluation_id
            )
            store.mark_reconciled(args.evaluation_id)
        except (OSError, ValueError, RuntimeError):
            store.fail(args.evaluation_id, "evaluation_failed")
            raise
    print(
        json.dumps(
            {
                "runtime_kind": result.runtime_kind,
                "gate_passed": result.real_model_runtime_gate_passed,
                "artifact_status": artifact.status.value,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from kagya.learning import (
    AdapterRegistry,
    BehavioralDimension,
    BehavioralEvaluationManifest,
    BehavioralEvaluator,
    BehavioralRuntimeKind,
    BehavioralScenario,
    BehavioralTrace,
    ExternalObservation,
    PublicBehaviorClass,
    ReproducibilityMetadata,
    fixture_set_hash,
    scenario_fixture_hash,
)
from kagya.learning.behavioral_evaluation import PairedBehavioralEvaluationResult
from kagya.learning.runtime_behavioral_runner import (
    deterministic_runtime_scenarios,
)
from kagya.learning.behavioral_coverage import BEHAVIORAL_COVERAGE_MANIFEST
from kagya.training.artifacts import sha256_file_map


def register_runtime_candidate(
    registry: AdapterRegistry, tmp_path: Path, adapter_id: str
):
    adapter_path = tmp_path / adapter_id
    adapter_path.mkdir(exist_ok=True)
    content = json.dumps({"adapter_id": adapter_id}, sort_keys=True).encode()
    (adapter_path / "adapter_config.json").write_bytes(content)
    adapter_hash = sha256_file_map({"adapter/adapter_config.json": content})
    return registry.register_candidate(
        adapter_id=adapter_id,
        adapter_path=adapter_path,
        dataset_path=tmp_path / f"{adapter_id}.jsonl",
        dataset_hash=adapter_id,
        base_model_revision="test-model-revision",
        adapter_hash=adapter_hash,
    )


def bind_runtime_behavioral_result(
    registry: AdapterRegistry,
    tmp_path: Path,
    adapter_id: str,
    *,
    passed: bool = True,
    evaluation_id: str | None = None,
    runtime_kind: BehavioralRuntimeKind = BehavioralRuntimeKind.DETERMINISTIC_RUNTIME,
) -> Path:
    result_path = write_runtime_behavioral_result(
        registry,
        tmp_path,
        adapter_id,
        passed=passed,
        evaluation_id=evaluation_id,
        runtime_kind=runtime_kind,
    )
    registry.apply_behavioral_evaluation(
        adapter_id,
        evaluation_id=evaluation_id or f"runtime-{adapter_id}",
        result_path=result_path,
    )
    registry.mark_behavioral_evaluation_reconciled(
        adapter_id, evaluation_id=evaluation_id or f"runtime-{adapter_id}"
    )
    return result_path


def write_runtime_behavioral_result(
    registry: AdapterRegistry,
    tmp_path: Path,
    adapter_id: str,
    *,
    passed: bool = True,
    evaluation_id: str | None = None,
    manifest_updates: dict[str, object] | None = None,
    runtime_kind: BehavioralRuntimeKind = BehavioralRuntimeKind.DETERMINISTIC_RUNTIME,
) -> Path:
    entry = registry.lookup(adapter_id)
    assert entry is not None
    evaluation_id = evaluation_id or f"runtime-{adapter_id}"
    scenario = BehavioralScenario(
        scenario_id="activation-gate",
        dimensions=(BehavioralDimension.IDENTITY_BOUNDARY,),
        initial_authoritative_state={"identity": {"origin": "self"}},
        observations=(
            ExternalObservation(sequence=1, event_type="probe", source="test-runtime"),
        ),
        expected_public_behavior=PublicBehaviorClass.NO_OP,
        reproducibility=ReproducibilityMetadata(
            subject_revision="test-subject-revision",
            fixture_revision="activation-v1",
            seed=133,
            clock="2026-07-23T00:00:00Z",
            runtime="test_runtime",
        ),
    )

    def baseline(_: BehavioralScenario) -> BehavioralTrace:
        return BehavioralTrace(
            final_authoritative_state=scenario.initial_authoritative_state,
            public_behavior=PublicBehaviorClass.NO_OP,
        )

    def candidate(_: BehavioralScenario) -> BehavioralTrace:
        return BehavioralTrace(
            final_authoritative_state=scenario.initial_authoritative_state,
            public_behavior=(
                PublicBehaviorClass.NO_OP if passed else PublicBehaviorClass.RESPOND
            ),
        )

    synthetic_id = f"source-{evaluation_id}"
    result = BehavioralEvaluator(tmp_path / "synthetic").evaluate_pair(
        synthetic_id,
        [scenario],
        baseline_id="baseline",
        baseline_runner=baseline,
        candidate_id=adapter_id,
        candidate_runner=candidate,
    )
    runtime_scenarios = deterministic_runtime_scenarios(
        subject_revision="test-subject-revision",
        runtime_kind=runtime_kind,
    )
    runtime_fixture_hashes = {
        item.scenario_id: scenario_fixture_hash(item) for item in runtime_scenarios
    }
    manifest_payload: dict[str, object] = {
        "source_commit_sha": "a" * 40,
        "subject_revision": "test-subject-revision",
        "runtime_schema_version": 1,
        "evaluator_schema_version": result.evaluator_version,
        "fixture_revision": runtime_scenarios[0].reproducibility.fixture_revision,
        "fixture_set_hash": fixture_set_hash(runtime_fixture_hashes),
        "config_hash": "b" * 64,
        "base_model_id": entry.base_model,
        "base_model_revision": entry.base_model_revision or "",
        "base_model_artifact_hash": "c" * 64,
        "candidate_adapter_id": adapter_id,
        "candidate_adapter_hash": entry.adapter_hash or "",
        "candidate_adapter_path_hash": entry.adapter_hash or "",
        "tool_registry_hash": "d" * 64,
        "policy_revision": "test-policy-v1",
        "state_schema_version": 1,
        "evaluator_implementation_hash": "e" * 64,
        "coverage_manifest_revision": BEHAVIORAL_COVERAGE_MANIFEST.revision,
        "coverage_manifest_hash": BEHAVIORAL_COVERAGE_MANIFEST.sha256,
    }
    manifest_payload.update(manifest_updates or {})
    manifest = BehavioralEvaluationManifest.model_validate(manifest_payload)
    payload = result.model_dump(mode="json")
    for subject_key in ("baseline", "candidate"):
        source_scenario = payload[subject_key]["scenario_results"][0]
        source_dimension = payload[subject_key]["dimension_scores"][0]
        payload[subject_key]["scenario_results"] = [
            {
                **source_scenario,
                "scenario_id": item.scenario_id,
                "dimensions": [dimension.value for dimension in item.dimensions],
                "runtime_kind": runtime_kind.value,
                "evaluated_hard_gates": (
                    [
                        requirement.hard_gate.value
                        for requirement in BEHAVIORAL_COVERAGE_MANIFEST.hard_gate_requirements
                        if requirement.required_scenario_id == item.scenario_id
                    ]
                ),
            }
            for item in runtime_scenarios
        ]
        dimensions = {
            dimension for item in runtime_scenarios for dimension in item.dimensions
        }
        payload[subject_key]["dimension_scores"] = [
            {
                **source_dimension,
                "dimension": dimension.value,
                "coverage_status": "passed" if passed else "failed",
            }
            for dimension in sorted(dimensions, key=lambda item: item.value)
        ]
    payload["fixture_hashes"] = runtime_fixture_hashes
    payload["reproducibility"] = {
        item.scenario_id: item.reproducibility.model_dump(mode="json")
        for item in runtime_scenarios
    }
    payload.update(
        {
            "evaluation_id": evaluation_id,
            "runtime_kind": runtime_kind.value,
            "deterministic_runtime_gate_passed": (
                passed
                if runtime_kind == BehavioralRuntimeKind.DETERMINISTIC_RUNTIME
                else False
            ),
            "real_model_runtime_gate_passed": (
                passed
                if runtime_kind == BehavioralRuntimeKind.REAL_MODEL_RUNTIME
                else False
            ),
            "manifest": manifest.model_dump(mode="json"),
            "coverage_complete": True,
            "missing_dimensions": [],
            "missing_hard_gates": [],
            "executed_scenarios": sorted(
                item.scenario_id for item in runtime_scenarios
            ),
            "coverage_manifest_revision": BEHAVIORAL_COVERAGE_MANIFEST.revision,
            "coverage_manifest_hash": BEHAVIORAL_COVERAGE_MANIFEST.sha256,
        }
    )
    runtime_result = PairedBehavioralEvaluationResult.model_validate(payload)
    result_path = tmp_path / f"{evaluation_id}.json"
    with result_path.open("x", encoding="utf-8") as output:
        json.dump(runtime_result.model_dump(mode="json"), output, sort_keys=True)
    return result_path

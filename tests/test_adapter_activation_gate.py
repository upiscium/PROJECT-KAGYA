import json
from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.learning import (
    ActivationEligibilityReason,
    AdapterRegistry,
    AdapterStatus,
)
from kagya.learning.behavioral_evaluation import PairedBehavioralEvaluationResult
from tests.adapter_behavioral_helpers import (
    bind_runtime_behavioral_result,
    register_runtime_candidate,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_unevaluated_behavioral_gate_fails_closed(tmp_path: Path) -> None:
    registry = _ordinary_approved(tmp_path)

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_UNEVALUATED
    with pytest.raises(ValueError, match="behavioral_unevaluated"):
        registry.activate("candidate")


def test_failed_behavioral_gate_is_distinct(tmp_path: Path) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate", passed=False)
    registry.approve("candidate")

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_FAILED


def test_valid_runtime_bound_behavioral_result_activates(tmp_path: Path) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    registry.approve("candidate")

    assert registry.activation_eligibility("candidate").eligible is True
    assert registry.activate("candidate").status == AdapterStatus.ACTIVE


def test_replaced_adapter_artifact_fails_hash_integrity(tmp_path: Path) -> None:
    registry = _ready(tmp_path)
    entry = registry.lookup("candidate")
    assert entry is not None
    (Path(entry.path) / "adapter_config.json").write_text("replaced", encoding="utf-8")

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.ADAPTER_ARTIFACT_MISMATCH
    assert "hash mismatch" in eligibility.detail


@pytest.mark.parametrize("passed", [True, False])
def test_ordinary_evaluation_preserves_behavioral_binding(
    tmp_path: Path, passed: bool
) -> None:
    registry = _registry(tmp_path)
    register_runtime_candidate(registry, tmp_path, "candidate")
    bind_runtime_behavioral_result(registry, tmp_path, "candidate", passed=passed)
    before = registry.lookup("candidate")

    after = registry.apply_evaluation(
        "candidate", score=0.9, result_path=tmp_path / "ordinary.json"
    )

    assert before is not None
    assert after.behavioral_evaluation_id == before.behavioral_evaluation_id
    assert after.behavioral_result_hash == before.behavioral_result_hash
    assert after.behavioral_gate_passed is passed
    assert after.candidate_adapter_hash == before.candidate_adapter_hash
    assert after.subject_revision == before.subject_revision
    assert after.fixture_set_hash == before.fixture_set_hash


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("deleted", ActivationEligibilityReason.BEHAVIORAL_RESULT_MISSING),
        ("corrupt", ActivationEligibilityReason.BEHAVIORAL_RESULT_CORRUPT),
        ("tampered", ActivationEligibilityReason.BEHAVIORAL_RESULT_TAMPERED),
    ],
)
def test_missing_corrupt_and_tampered_results_are_distinct(
    tmp_path: Path,
    mutation: str,
    reason: ActivationEligibilityReason,
) -> None:
    registry = _ready(tmp_path)
    entry = registry.lookup("candidate")
    assert entry is not None and entry.behavioral_evaluation_path is not None
    path = Path(entry.behavioral_evaluation_path)
    if mutation == "deleted":
        path.unlink()
    elif mutation == "corrupt":
        path.write_text("{", encoding="utf-8")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["candidate"]["aggregate_score"] = 0.5
        path.write_text(json.dumps(payload), encoding="utf-8")

    assert registry.activation_eligibility("candidate").reason == reason


def test_candidate_id_mismatch_fails_closed_at_activation(tmp_path: Path) -> None:
    registry = _ready(tmp_path)
    entry = registry.lookup("candidate")
    assert entry is not None and entry.behavioral_evaluation_path is not None
    path = Path(entry.behavioral_evaluation_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate"]["subject_id"] = "different-candidate"
    payload["adapter_binding"]["candidate_adapter_id"] = "different-candidate"
    validated = PairedBehavioralEvaluationResult.model_validate(payload)
    path.write_text(json.dumps(validated.model_dump(mode="json")), encoding="utf-8")

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_BINDING_MISMATCH
    assert "candidate ID mismatch" in eligibility.detail


def test_candidate_hash_mismatch_is_distinct(tmp_path: Path) -> None:
    registry = _ready(tmp_path)
    _update_registry_field(registry, "adapter_hash", "different-adapter-hash")

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_BINDING_MISMATCH
    assert "candidate adapter hash mismatch" in eligibility.detail


def test_stale_registry_binding_is_distinct(tmp_path: Path) -> None:
    registry = _ready(tmp_path)
    _update_registry_field(registry, "fixture_set_hash", "0" * 64)

    assert (
        registry.activation_eligibility("candidate").reason
        == ActivationEligibilityReason.BEHAVIORAL_RESULT_STALE
    )


def test_result_schema_failure_is_distinct_from_corrupt_json(tmp_path: Path) -> None:
    registry = _ready(tmp_path)
    entry = registry.lookup("candidate")
    assert entry is not None and entry.behavioral_evaluation_path is not None
    path = Path(entry.behavioral_evaluation_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["candidate"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        registry.activation_eligibility("candidate").reason
        == ActivationEligibilityReason.BEHAVIORAL_RESULT_SCHEMA_INVALID
    )


def test_legacy_activation_boolean_never_migrates_as_behavioral_authority(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.path.write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "adapter_id": "legacy",
                        "base_model": "model",
                        "path": str(tmp_path / "legacy"),
                        "status": "approved",
                        "dataset_path": str(tmp_path / "dataset.jsonl"),
                        "dataset_hash": "dataset",
                        "activation_gate_passed": True,
                        "behavioral_gate_passed": True,
                        "schema_version": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entry = registry.lookup("legacy")

    assert entry is not None
    assert entry.behavioral_gate_passed is None
    assert entry.activation_gate_passed is False
    assert entry.schema_version == 4
    assert (
        registry.activation_eligibility("legacy").reason
        == ActivationEligibilityReason.BEHAVIORAL_UNEVALUATED
    )


def test_legacy_active_adapter_keeps_running_with_warning(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.path.write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "adapter_id": "legacy-active",
                        "base_model": "model",
                        "path": str(tmp_path / "legacy"),
                        "status": "active",
                        "dataset_path": str(tmp_path / "dataset.jsonl"),
                        "dataset_hash": "dataset",
                        "activation_gate_passed": True,
                        "schema_version": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="may keep running"):
        entry = registry.lookup("legacy-active")

    assert entry is not None
    assert entry.status == AdapterStatus.ACTIVE
    assert entry.legacy_activation_warning is True


def _ready(tmp_path: Path) -> AdapterRegistry:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    registry.approve("candidate")
    return registry


def _ordinary_approved(tmp_path: Path) -> AdapterRegistry:
    registry = _ordinary_evaluated(tmp_path)
    registry.approve("candidate")
    return registry


def _ordinary_evaluated(tmp_path: Path) -> AdapterRegistry:
    registry = _registry(tmp_path)
    register_runtime_candidate(registry, tmp_path, "candidate")
    registry.apply_evaluation(
        "candidate", score=0.9, result_path=tmp_path / "ordinary.json"
    )
    return registry


def _registry(tmp_path: Path) -> AdapterRegistry:
    settings = load_settings(CONFIG_PATH)
    return AdapterRegistry(
        settings.model_copy(
            update={
                "adapter_registry": settings.adapter_registry.model_copy(
                    update={
                        "path": tmp_path / "registry.json",
                        "eval_result_dir": tmp_path / "eval-results",
                        "eval_sets": [],
                    }
                )
            }
        )
    )


def _update_registry_field(registry: AdapterRegistry, field: str, value: str) -> None:
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    payload["adapters"][0][field] = value
    registry.path.write_text(json.dumps(payload), encoding="utf-8")

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kagya._build_info import SourceRevisionStatus
from kagya.artifact_provenance import build_model_artifact_manifest
from kagya.config import BehavioralActivationPolicy, ProjectEnvironment, load_settings
from kagya.learning import (
    ActivationEligibilityReason,
    AdapterRegistry,
    AdapterStatus,
    BehavioralEvaluationManifest,
    BehavioralRuntimeKind,
)
from kagya.learning.behavioral_evaluation import PairedBehavioralEvaluationResult
from kagya.learning.behavioral_coverage import BEHAVIORAL_COVERAGE_MANIFEST
from kagya.learning.runtime_behavioral_runner import current_evaluator_hash
from kagya.training.artifacts import sha256_file_map
from tests.adapter_behavioral_helpers import (
    bind_runtime_behavioral_result,
    register_runtime_candidate,
    write_runtime_behavioral_result,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_unevaluated_behavioral_gate_fails_closed(tmp_path: Path) -> None:
    registry = _ordinary_approved(tmp_path)

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_UNEVALUATED
    with pytest.raises(ValueError, match="behavioral_unevaluated"):
        registry.activate("candidate")


def test_disabled_behavioral_policy_does_not_bypass_identity_integrity(
    tmp_path: Path,
) -> None:
    registry = _ordinary_approved(tmp_path)
    settings = registry.settings.model_copy(
        update={
            "adapter_registry": registry.settings.adapter_registry.model_copy(
                update={
                    "behavioral_activation_policy": BehavioralActivationPolicy.DISABLED
                }
            )
        }
    )
    disabled = AdapterRegistry(settings)

    assert (
        disabled.activation_eligibility("candidate").reason
        == ActivationEligibilityReason.IDENTITY_NOT_EVALUATED
    )
    with pytest.raises(ValueError, match="identity_not_evaluated"):
        disabled.activate("candidate")


def test_failed_behavioral_gate_is_distinct(tmp_path: Path) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate", passed=False)
    registry.approve("candidate")

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_FAILED


def test_valid_runtime_bound_behavioral_result_activates(tmp_path: Path) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    bound = registry.lookup("candidate")
    assert bound is not None
    assert bound.behavioral_candidate_adapter_hash == bound.adapter_hash
    assert bound.behavioral_base_model_revision == bound.base_model_revision
    registry.approve("candidate")

    assert registry.activation_eligibility("candidate").eligible is True
    assert registry.activate("candidate").status == AdapterStatus.ACTIVE


def test_identity_assessment_missing_failed_and_stale_fail_closed(
    tmp_path: Path,
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    registry.approve("candidate")
    payload = json.loads(registry.path.read_text(encoding="utf-8"))

    payload["adapters"][0]["identity_drift_assessment"] = None
    registry.path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        registry.activation_eligibility("candidate").reason
        == ActivationEligibilityReason.IDENTITY_NOT_EVALUATED
    )

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed = _ordinary_evaluated(failed_root)
    bind_runtime_behavioral_result(failed, failed_root, "candidate", passed=False)
    failed.approve("candidate")
    assert failed.identity_assessment_status("candidate")[0] == "failed"
    assert failed.activation_eligibility("candidate").eligible is False

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    stale = _ready(stale_root)
    stale_payload = json.loads(stale.path.read_text(encoding="utf-8"))
    stale_payload["adapters"][0]["identity_drift_assessment"]["adapter_hash"] = "0" * 64
    stale.path.write_text(json.dumps(stale_payload), encoding="utf-8")
    assert (
        stale.activation_eligibility("candidate").reason
        == ActivationEligibilityReason.IDENTITY_STALE
    )


def test_production_policy_rejects_missing_real_model_evidence(tmp_path: Path) -> None:
    registry = _ready(tmp_path)
    required = _with_real_model_policy(registry)

    eligibility = required.activation_eligibility("candidate")

    assert eligibility.reason == ActivationEligibilityReason.REAL_MODEL_NOT_RUN
    with pytest.raises(ValueError, match="real_model_not_run"):
        required.activate("candidate")


def test_current_real_model_pass_allows_activation(tmp_path: Path) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        evaluation_id="real-pass",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    registry.approve("candidate")
    required = _with_real_model_policy(registry)

    assert required.activation_eligibility("candidate").eligible is True
    assert required.activate("candidate").status == AdapterStatus.ACTIVE


def test_development_deterministic_policy_is_explicit_and_real_policy_is_distinct(
    tmp_path: Path,
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    registry.approve("candidate")

    optional = registry.activation_eligibility("candidate")
    assert optional.eligible is True
    assert optional.real_model_required is False
    assert optional.real_model_status == "not_run"

    required_settings = registry.settings.model_copy(
        update={
            "adapter_registry": registry.settings.adapter_registry.model_copy(
                update={
                    "behavioral_activation_policy": BehavioralActivationPolicy.REAL_MODEL_REQUIRED
                }
            )
        }
    )
    required = AdapterRegistry(required_settings).activation_eligibility("candidate")
    assert required.eligible is False
    assert required.real_model_required is True
    assert required.reason == ActivationEligibilityReason.REAL_MODEL_NOT_RUN


def test_required_real_model_gate_reports_failed_and_stale_distinctly(
    tmp_path: Path,
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        passed=False,
        evaluation_id="real-failed",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    required_settings = registry.settings.model_copy(
        update={
            "adapter_registry": registry.settings.adapter_registry.model_copy(
                update={
                    "behavioral_activation_policy": BehavioralActivationPolicy.REAL_MODEL_REQUIRED
                }
            )
        }
    )
    required_registry = AdapterRegistry(required_settings)
    assert (
        required_registry.activation_eligibility("candidate").reason
        == ActivationEligibilityReason.REAL_MODEL_FAILED
    )

    bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        evaluation_id="real-passed",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    _update_registry_field(registry, "real_model_fixture_set_hash", "f" * 64)
    assert (
        required_registry.activation_eligibility("candidate").reason
        == ActivationEligibilityReason.REAL_MODEL_STALE
    )


@pytest.mark.parametrize("mutation", ["adapter_content", "result_hash"])
def test_real_model_activation_revalidates_current_files(
    tmp_path: Path, mutation: str
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    result_path = bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        evaluation_id="real-current-files",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    registry.approve("candidate")
    required_settings = registry.settings.model_copy(
        update={
            "adapter_registry": registry.settings.adapter_registry.model_copy(
                update={
                    "behavioral_activation_policy": BehavioralActivationPolicy.REAL_MODEL_REQUIRED
                }
            )
        }
    )
    required_registry = AdapterRegistry(required_settings)
    assert required_registry.activation_eligibility("candidate").eligible is True

    entry = registry.lookup("candidate")
    assert entry is not None
    if mutation == "adapter_content":
        (Path(entry.path) / "adapter_config.json").write_text(
            '{"adapter_id":"replaced"}', encoding="utf-8"
        )
    else:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["created_at"] = "2026-07-24T00:00:00Z"
        result_path.write_text(json.dumps(payload), encoding="utf-8")

    eligibility = required_registry.activation_eligibility("candidate")
    assert eligibility.eligible is False
    assert eligibility.real_model_status == "hash_mismatch"
    assert eligibility.reason == (
        ActivationEligibilityReason.ADAPTER_ARTIFACT_MISMATCH
        if mutation == "adapter_content"
        else ActivationEligibilityReason.REAL_MODEL_HASH_MISMATCH
    )


def test_failed_real_model_gate_still_revalidates_corrupt_result(
    tmp_path: Path,
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    result_path = bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        passed=False,
        evaluation_id="real-failed-corrupt",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    result_path.write_text("{", encoding="utf-8")
    required_settings = registry.settings.model_copy(
        update={
            "adapter_registry": registry.settings.adapter_registry.model_copy(
                update={
                    "behavioral_activation_policy": BehavioralActivationPolicy.REAL_MODEL_REQUIRED
                }
            )
        }
    )

    eligibility = AdapterRegistry(required_settings).activation_eligibility("candidate")

    assert eligibility.real_model_status == "corrupt"
    assert eligibility.reason == ActivationEligibilityReason.REAL_MODEL_CORRUPT


def test_real_model_gate_reports_incomplete_canonical_coverage(
    tmp_path: Path,
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    result_path = bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        evaluation_id="real-incomplete",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["candidate"]["scenario_results"].pop()
    validated = PairedBehavioralEvaluationResult.model_validate(payload)
    content = json.dumps(validated.model_dump(mode="json"), sort_keys=True).encode()
    result_path.write_bytes(content)
    _update_registry_field(
        registry,
        "real_model_behavioral_result_hash",
        hashlib.sha256(content).hexdigest(),
    )

    eligibility = _with_real_model_policy(registry).activation_eligibility("candidate")

    assert eligibility.real_model_status == "coverage_incomplete"
    assert (
        eligibility.reason == ActivationEligibilityReason.REAL_MODEL_COVERAGE_INCOMPLETE
    )


@pytest.mark.parametrize(
    "missing_id",
    sorted(
        {
            scenario_id
            for requirement in BEHAVIORAL_COVERAGE_MANIFEST.requirements
            for scenario_id in requirement.required_scenario_ids
        }
    ),
)
def test_each_deleted_runtime_scenario_rejects_activation(
    tmp_path: Path, missing_id: str
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    result_path = bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    registry.approve("candidate")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["candidate"]["scenario_results"] = [
        item
        for item in payload["candidate"]["scenario_results"]
        if item["scenario_id"] != missing_id
    ]
    validated = PairedBehavioralEvaluationResult.model_validate(payload)
    content = json.dumps(validated.model_dump(mode="json"), sort_keys=True).encode()
    result_path.write_bytes(content)
    _update_registry_field(
        registry,
        "behavioral_result_hash",
        hashlib.sha256(content).hexdigest(),
    )

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.eligible is False
    assert (
        eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_COVERAGE_INCOMPLETE
    )


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
    assert (
        after.behavioral_candidate_adapter_hash
        == before.behavioral_candidate_adapter_hash
    )
    assert after.behavioral_base_model_revision == before.behavioral_base_model_revision
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
    payload["manifest"]["candidate_adapter_id"] = "different-candidate"
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_adapter_hash", "0" * 64, "candidate adapter hash mismatch"),
        ("base_model_id", "different-model", "base model mismatch"),
        ("base_model_revision", "different-revision", "revision mismatch"),
        ("candidate_adapter_path_hash", "0" * 64, "path hash mismatch"),
    ],
)
def test_registry_and_artifact_manifest_mismatches_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    result_path = write_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        manifest_updates={field: value},
    )

    with pytest.raises(ValueError, match=message):
        registry.apply_behavioral_evaluation(
            "candidate",
            evaluation_id="runtime-candidate",
            result_path=result_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("subject_revision", "different-subject", "subject revision mismatch"),
        ("fixture_revision", "different-fixtures", "fixture revision mismatch"),
        ("fixture_set_hash", "0" * 64, "fixture set hash mismatch"),
        ("evaluator_schema_version", 2, "evaluator schema version mismatch"),
    ],
)
def test_result_manifest_identity_mismatches_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    registry = _ordinary_evaluated(tmp_path)

    with pytest.raises(ValidationError, match=message):
        write_runtime_behavioral_result(
            registry,
            tmp_path,
            "candidate",
            manifest_updates={field: value},
        )


def test_runtime_manifest_requires_every_identity_field(tmp_path: Path) -> None:
    manifest_payload = _manifest_payload(tmp_path)

    for field in BehavioralEvaluationManifest.model_fields:
        if field == "schema_version":
            continue
        incomplete = dict(manifest_payload)
        del incomplete[field]
        with pytest.raises(ValidationError, match=field):
            BehavioralEvaluationManifest.model_validate(incomplete)


def test_runtime_manifest_is_immutable(tmp_path: Path) -> None:
    manifest = BehavioralEvaluationManifest.model_validate(_manifest_payload(tmp_path))

    with pytest.raises(ValidationError, match="frozen"):
        manifest.policy_revision = "changed"


@pytest.mark.parametrize(
    "field",
    [
        "config_hash",
        "base_model_artifact_hash",
        "candidate_adapter_hash",
        "candidate_adapter_path_hash",
        "tool_registry_hash",
        "evaluator_implementation_hash",
    ],
)
def test_runtime_manifest_rejects_non_sha256_hashes(tmp_path: Path, field: str) -> None:
    manifest_payload = _manifest_payload(tmp_path)
    manifest_payload[field] = "ABC123"

    with pytest.raises(ValidationError, match=field):
        BehavioralEvaluationManifest.model_validate(manifest_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit_sha", "not-a-commit"),
        ("runtime_schema_version", 0),
        ("policy_revision", ""),
        ("state_schema_version", 0),
    ],
)
def test_runtime_manifest_rejects_invalid_non_hash_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest_payload = _manifest_payload(tmp_path)
    manifest_payload[field] = value

    with pytest.raises(ValidationError, match=field):
        BehavioralEvaluationManifest.model_validate(manifest_payload)


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
    del payload["manifest"]
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
    assert entry.schema_version == 11
    assert (
        json.loads(registry.path.read_text(encoding="utf-8"))["adapters"][0][
            "schema_version"
        ]
        == 11
    )
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


def test_schema_v4_behavioral_fields_migrate_to_exact_names(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.path.write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "adapter_id": "v4",
                        "base_model": "model",
                        "base_model_revision": "model-revision",
                        "path": str(tmp_path / "v4"),
                        "status": "approved",
                        "dataset_path": str(tmp_path / "dataset.jsonl"),
                        "dataset_hash": "dataset",
                        "quality_gate_passed": True,
                        "holdout_gate_passed": True,
                        "drift_gate_passed": True,
                        "behavioral_gate_passed": True,
                        "behavioral_evaluation_id": "old-runtime",
                        "candidate_adapter_hash": "f" * 64,
                        "schema_version": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entry = registry.lookup("v4")

    assert entry is not None
    assert entry.schema_version == 11
    assert entry.behavioral_candidate_adapter_hash == "f" * 64
    assert entry.behavioral_base_model_revision == "model-revision"


def test_schema_v7_real_model_result_does_not_migrate_as_authority(
    tmp_path: Path,
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        evaluation_id="legacy-real",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    payload["adapters"][0]["schema_version"] = 7
    registry.path.write_text(json.dumps(payload), encoding="utf-8")

    eligibility = _with_real_model_policy(registry).activation_eligibility("candidate")

    assert eligibility.real_model_status == "not_run"
    assert (
        eligibility.reason == ActivationEligibilityReason.BEHAVIORAL_COVERAGE_INCOMPLETE
    )


def test_schema_v9_migration_persists_v11_and_accepts_new_real_evidence(
    tmp_path: Path,
) -> None:
    registry = _ordinary_evaluated(tmp_path)
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    payload["adapters"][0]["schema_version"] = 9
    registry.path.write_text(json.dumps(payload), encoding="utf-8")

    migrated = AdapterRegistry(registry.settings)
    entry = migrated.lookup("candidate")
    assert entry is not None and entry.schema_version == 11
    assert (
        json.loads(migrated.path.read_text(encoding="utf-8"))["adapters"][0][
            "schema_version"
        ]
        == 11
    )

    bind_runtime_behavioral_result(
        migrated,
        tmp_path,
        "candidate",
        evaluation_id="new-real-after-v9",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    rebound = migrated.lookup("candidate")
    assert rebound is not None
    assert rebound.real_model_behavioral_evaluation_id == "new-real-after-v9"
    assert rebound.real_model_behavioral_artifact_state == "reconciled"


def test_production_full_eligibility_requires_coherent_deterministic_and_real_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_commit = "1" * 40
    processor_commit = "2" * 40
    source_commit = "3" * 40
    source_tree = "4" * 40
    base = load_settings(CONFIG_PATH)
    settings = base.model_copy(
        update={
            "project": base.project.model_copy(
                update={"environment": ProjectEnvironment.PRODUCTION}
            ),
            "model": base.model.model_copy(
                update={
                    "primary_id": "production-model",
                    "revision": model_commit,
                    "processor_revision": processor_commit,
                }
            ),
            "adapter_registry": base.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "registry.json",
                    "eval_result_dir": tmp_path / "evaluations",
                    "eval_sets": [],
                    "behavioral_activation_policy": BehavioralActivationPolicy.REAL_MODEL_REQUIRED,
                }
            ),
        }
    )
    adapter = tmp_path / "candidate"
    adapter.mkdir()
    config = b'{"peft_type":"LORA"}'
    (adapter / "adapter_config.json").write_bytes(config)
    registry = AdapterRegistry(settings)
    registry.register_candidate(
        adapter_id="candidate",
        adapter_path=adapter,
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="dataset",
        base_model="production-model",
        base_model_revision=model_commit,
        adapter_hash=sha256_file_map({"adapter/adapter_config.json": config}),
    )
    registry.apply_evaluation("candidate", score=0.9, result_path=tmp_path / "ordinary")
    common = {
        "source_commit_sha": source_commit,
        "source_revision_status": "verified",
        "source_tree_hash": source_tree,
        "base_model_revision_requested": model_commit,
        "processor_revision_requested": processor_commit,
    }
    bind_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        manifest_updates={
            **common,
            "evaluator_implementation_hash": current_evaluator_hash(
                BehavioralRuntimeKind.DETERMINISTIC_RUNTIME
            ),
        },
    )
    model_snapshot = tmp_path / "model-snapshot"
    processor_snapshot = tmp_path / "processor-snapshot"
    model_snapshot.mkdir()
    processor_snapshot.mkdir()
    (model_snapshot / "config.json").write_bytes(b"{}")
    (model_snapshot / "model.safetensors").write_bytes(b"weights")
    (processor_snapshot / "tokenizer.json").write_bytes(b"{}")
    model_manifest = build_model_artifact_manifest(
        model_snapshot,
        processor_snapshot=processor_snapshot,
        model_id="production-model",
        requested_revision=model_commit,
        resolved_revision=model_commit,
        processor_requested_revision=processor_commit,
        processor_resolved_revision=processor_commit,
    )
    real_path = write_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        evaluation_id="production-real",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
        manifest_updates={
            **common,
            "base_model_revision_resolved": model_commit,
            "processor_revision_resolved": processor_commit,
            "model_artifact_manifest_hash": model_manifest.sha256,
            "model_artifact_manifest": model_manifest.model_dump(mode="json"),
            "evaluator_implementation_hash": current_evaluator_hash(
                BehavioralRuntimeKind.REAL_MODEL_RUNTIME
            ),
        },
    )
    real_payload = json.loads(real_path.read_text(encoding="utf-8"))
    real_payload["baseline_generation_count"] = 1
    real_payload["candidate_generation_count"] = 1
    real_path.write_text(json.dumps(real_payload), encoding="utf-8")
    registry.apply_behavioral_evaluation(
        "candidate", evaluation_id="production-real", result_path=real_path
    )
    registry.mark_behavioral_evaluation_reconciled(
        "candidate", evaluation_id="production-real"
    )
    registry.approve("candidate")
    monkeypatch.setattr(
        "kagya._build_info.resolve_source_build_info",
        lambda: SimpleNamespace(
            status=SourceRevisionStatus.VERIFIED,
            commit_sha=source_commit,
            tree_hash=source_tree,
        ),
    )
    monkeypatch.setattr(
        "kagya.learning.adapter_registry._local_model_snapshot",
        lambda _model_id, revision: (
            model_snapshot if revision == model_commit else processor_snapshot
        ),
    )

    eligibility = registry.activation_eligibility("candidate")

    assert eligibility.eligible is True

    entry = registry.lookup("candidate")
    assert entry is not None and entry.behavioral_evaluation_path is not None
    deterministic_path = Path(entry.behavioral_evaluation_path)
    deterministic_payload = json.loads(deterministic_path.read_text(encoding="utf-8"))
    deterministic_payload["manifest"]["processor_revision_requested"] = "5" * 40
    deterministic_path.write_text(json.dumps(deterministic_payload), encoding="utf-8")
    registry_payload = json.loads(registry.path.read_text(encoding="utf-8"))
    registry_payload["adapters"][0]["behavioral_result_hash"] = hashlib.sha256(
        deterministic_path.read_bytes()
    ).hexdigest()
    registry.path.write_text(json.dumps(registry_payload), encoding="utf-8")

    mismatch = AdapterRegistry(settings).activation_eligibility("candidate")
    assert mismatch.eligible is False
    assert mismatch.reason == ActivationEligibilityReason.MODEL_PROVENANCE_MISMATCH


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


def _with_real_model_policy(registry: AdapterRegistry) -> AdapterRegistry:
    settings = registry.settings.model_copy(
        update={
            "adapter_registry": registry.settings.adapter_registry.model_copy(
                update={
                    "behavioral_activation_policy": BehavioralActivationPolicy.REAL_MODEL_REQUIRED
                }
            )
        }
    )
    return AdapterRegistry(settings)


def _update_registry_field(registry: AdapterRegistry, field: str, value: str) -> None:
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    payload["adapters"][0][field] = value
    registry.path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest_payload(tmp_path: Path) -> dict[str, object]:
    registry = _ordinary_evaluated(tmp_path)
    path = write_runtime_behavioral_result(registry, tmp_path, "candidate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["manifest"])

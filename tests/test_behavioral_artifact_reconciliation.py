import hashlib
import json
from pathlib import Path

import pytest

from kagya.learning import (
    AdapterRegistry,
    BehavioralArtifactStatus,
    BehavioralArtifactStore,
)
from kagya.api.routes.evaluations import list_behavioral_evaluations
from kagya.config import load_settings
from tests.adapter_behavioral_helpers import (
    register_runtime_candidate,
    write_runtime_behavioral_result,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_artifact_commit_is_finalized_valid_and_relative(tmp_path: Path) -> None:
    store = BehavioralArtifactStore(tmp_path)

    record = store.commit("runtime-one", {"evaluation_id": "runtime-one"})

    assert record.status == BehavioralArtifactStatus.VALID
    assert record.relative_path == "behavioral/runtime-one.json"
    assert not Path(record.relative_path).is_absolute()
    assert store.valid("runtime-one") is not None


def test_reconciliation_classifies_orphans_hash_mismatch_and_corruption(
    tmp_path: Path,
) -> None:
    store = BehavioralArtifactStore(tmp_path)
    store.commit("hash-mismatch", {"evaluation_id": "hash-mismatch"})
    mismatch_path = tmp_path / "behavioral" / "hash-mismatch.json"
    mismatch_path.write_text('{"evaluation_id":"changed"}', encoding="utf-8")
    corrupt_path = tmp_path / "behavioral" / "corrupt.json"
    corrupt_path.write_text("{corrupt", encoding="utf-8")
    orphan_path = tmp_path / "behavioral" / "orphan.json"
    orphan_path.write_text('{"evaluation_id":"orphan"}', encoding="utf-8")
    registry = json.loads(store.registry_path.read_text(encoding="utf-8"))
    missing_content = b'{"evaluation_id":"missing"}'
    registry["artifacts"].append(
        {
            "evaluation_id": "missing",
            "relative_path": "behavioral/missing.json",
            "sha256": hashlib.sha256(missing_content).hexdigest(),
            "status": "valid",
            "updated_at": "2026-07-23T00:00:00Z",
        }
    )
    store.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    statuses = {item.evaluation_id: item.status for item in store.reconcile()}

    assert statuses["hash-mismatch"] == BehavioralArtifactStatus.HASH_MISMATCH
    assert statuses["corrupt"] == BehavioralArtifactStatus.CORRUPT
    assert statuses["orphan"] == BehavioralArtifactStatus.ORPHAN_RESULT
    assert statuses["missing"] == BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE


def test_prepared_transaction_remains_reconcilable(tmp_path: Path) -> None:
    store = BehavioralArtifactStore(tmp_path)
    prepared = tmp_path / "behavioral" / ".prepared-one.json.prepared"
    prepared.parent.mkdir(parents=True)
    content = b'{"evaluation_id":"prepared-one"}'
    prepared.write_bytes(content)
    store.registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "evaluation_id": "prepared-one",
                        "relative_path": "behavioral/prepared-one.json",
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "status": "prepared",
                        "updated_at": "2026-07-23T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = store.reconcile()

    assert records[0].status == BehavioralArtifactStatus.PREPARED

    prepared.write_text("{corrupt", encoding="utf-8")
    assert store.reconcile()[0].status == BehavioralArtifactStatus.CORRUPT


def test_corrupt_artifact_does_not_break_behavioral_history(tmp_path: Path) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "eval_result_dir": tmp_path / "results",
                    "path": tmp_path / "adapters.json",
                }
            )
        }
    )
    registry = AdapterRegistry(settings)
    register_runtime_candidate(registry, tmp_path, "candidate")
    source = write_runtime_behavioral_result(
        registry, tmp_path, "candidate", evaluation_id="valid-history"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    store.prepare("valid-history", payload)
    registry.prepare_behavioral_evaluation(
        "candidate",
        evaluation_id="valid-history",
        prepared_path=store.prepared_path("valid-history"),
        final_path=store.final_path("valid-history"),
    )
    store.finalize("valid-history")
    registry.finalize_behavioral_evaluation("candidate", evaluation_id="valid-history")
    (store.behavioral_dir / "broken.json").write_text("{broken", encoding="utf-8")

    response = list_behavioral_evaluations(settings, registry)

    summaries = {item.evaluation_id: item for item in response.results}
    assert summaries["valid-history"].artifact_status == "valid"
    assert summaries["broken"].artifact_status == "corrupt"
    assert summaries["broken"].quarantine_error == (
        "Artifact quarantined by integrity reconciliation"
    )
    assert str(tmp_path) not in summaries["broken"].model_dump_json()


def test_cross_registry_reconciliation_is_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "eval_result_dir": tmp_path / "results",
                    "path": tmp_path / "adapters.json",
                }
            )
        }
    )
    registry = AdapterRegistry(settings)
    register_runtime_candidate(registry, tmp_path, "candidate")
    source = write_runtime_behavioral_result(
        registry, tmp_path, "candidate", evaluation_id="cross-registry"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)

    store.prepare("cross-registry", payload)
    assert store.reconcile(registry)[0].status == BehavioralArtifactStatus.PREPARED
    registry.prepare_behavioral_evaluation(
        "candidate",
        evaluation_id="cross-registry",
        prepared_path=store.prepared_path("cross-registry"),
        final_path=store.final_path("cross-registry"),
    )
    first = store.reconcile(registry)
    second = store.reconcile(registry)
    assert [item.status for item in first] == [BehavioralArtifactStatus.PREPARED]
    assert [item.status for item in second] == [BehavioralArtifactStatus.PREPARED]
    assert registry.activation_eligibility("candidate").eligible is False

    registry.finalize_behavioral_evaluation("candidate", evaluation_id="cross-registry")
    assert store.reconcile(registry)[0].status == BehavioralArtifactStatus.VALID

    store.final_path("cross-registry").unlink()
    assert store.reconcile(registry)[0].status == (
        BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE
    )
    assert registry.activation_eligibility("candidate").eligible is False


def test_cross_registry_reconciliation_classifies_orphan_and_binding_mismatch(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "eval_result_dir": tmp_path / "results",
                    "path": tmp_path / "adapters.json",
                }
            )
        }
    )
    registry = AdapterRegistry(settings)
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    store.commit("unbound", {"evaluation_id": "unbound"})
    assert store.reconcile(registry)[0].status == BehavioralArtifactStatus.ORPHAN_RESULT

    register_runtime_candidate(registry, tmp_path, "candidate")
    source = write_runtime_behavioral_result(
        registry, tmp_path, "candidate", evaluation_id="mismatch"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    store.prepare("mismatch", payload)
    registry.prepare_behavioral_evaluation(
        "candidate",
        evaluation_id="mismatch",
        prepared_path=store.prepared_path("mismatch"),
        final_path=store.final_path("mismatch"),
    )
    store.finalize("mismatch")
    registry.finalize_behavioral_evaluation("candidate", evaluation_id="mismatch")
    raw = json.loads(registry.path.read_text(encoding="utf-8"))
    raw["adapters"][0]["behavioral_result_hash"] = "0" * 64
    registry.path.write_text(json.dumps(raw), encoding="utf-8")

    statuses = {item.evaluation_id: item.status for item in store.reconcile(registry)}
    assert statuses["unbound"] == BehavioralArtifactStatus.ORPHAN_RESULT
    assert statuses["mismatch"] == BehavioralArtifactStatus.HASH_MISMATCH
    first = store.reconcile(registry, quarantine_invalid=True)
    second = store.reconcile(registry, quarantine_invalid=True)
    assert first == second
    quarantined = registry.lookup("candidate")
    assert quarantined is not None
    assert quarantined.behavioral_artifact_state == "quarantined"
    assert quarantined.behavioral_gate_passed is False


@pytest.mark.parametrize(
    ("boundary", "expected_status", "expected_binding_state"),
    (
        ("prepared", BehavioralArtifactStatus.PREPARED, "unbound"),
        ("bound", BehavioralArtifactStatus.PREPARED, "prepared"),
        ("artifact_finalized", BehavioralArtifactStatus.PREPARED, "prepared"),
        ("registry_finalized", BehavioralArtifactStatus.VALID, "finalized"),
    ),
)
def test_behavioral_saga_crash_boundaries_never_grant_activation(
    tmp_path: Path,
    boundary: str,
    expected_status: BehavioralArtifactStatus,
    expected_binding_state: str,
) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "eval_result_dir": tmp_path / "results",
                    "path": tmp_path / "adapters.json",
                }
            )
        }
    )
    registry = AdapterRegistry(settings)
    register_runtime_candidate(registry, tmp_path, "candidate")
    source = write_runtime_behavioral_result(
        registry, tmp_path, "candidate", evaluation_id="saga"
    )
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    store.prepare("saga", json.loads(source.read_text(encoding="utf-8")))
    if boundary != "prepared":
        registry.prepare_behavioral_evaluation(
            "candidate",
            evaluation_id="saga",
            prepared_path=store.prepared_path("saga"),
            final_path=store.final_path("saga"),
        )
    if boundary in {"artifact_finalized", "registry_finalized"}:
        store.finalize("saga")
    if boundary == "registry_finalized":
        registry.finalize_behavioral_evaluation("candidate", evaluation_id="saga")

    restarted_store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    restarted_registry = AdapterRegistry(settings)
    record = restarted_store.reconcile(restarted_registry)[0]
    entry = restarted_registry.lookup("candidate")
    assert entry is not None
    assert record.status == expected_status
    assert entry.behavioral_artifact_state == expected_binding_state
    assert restarted_registry.activation_eligibility("candidate").eligible is False

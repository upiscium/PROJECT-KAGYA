import hashlib
import json
from pathlib import Path

from kagya.learning import BehavioralArtifactStatus, BehavioralArtifactStore
from kagya.api.routes.evaluations import list_behavioral_evaluations
from kagya.config import load_settings


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


def test_corrupt_artifact_does_not_break_behavioral_history(tmp_path: Path) -> None:
    store = BehavioralArtifactStore(tmp_path)
    store.commit(
        "valid-history",
        {
            "evaluation_id": "valid-history",
            "created_at": "2026-07-23T00:00:00Z",
            "baseline": {
                "subject_id": "baseline",
                "aggregate_score": 1.0,
                "dimension_scores": [],
            },
            "candidate": {
                "subject_id": "candidate",
                "aggregate_score": 1.0,
                "dimension_scores": [],
                "hard_gate_failures": [],
            },
        },
    )
    (tmp_path / "behavioral" / "broken.json").write_text("{broken", encoding="utf-8")
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"eval_result_dir": tmp_path}
            )
        }
    )

    response = list_behavioral_evaluations(settings)

    assert [item.evaluation_id for item in response.results] == ["valid-history"]

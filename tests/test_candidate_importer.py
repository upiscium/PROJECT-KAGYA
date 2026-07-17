from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.learning import AdapterRegistry, AdapterStatus
from kagya.training import (
    CandidateArtifactImporter,
    TrainingArtifactContract,
    TrainingBundleManifest,
    TrainingResultManifest,
    sha256_bytes,
    sha256_file_map,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_valid_result_imports_local_candidate_idempotently(tmp_path: Path) -> None:
    settings, bundle, result = _artifacts(tmp_path)
    registry = AdapterRegistry(settings)
    importer = CandidateArtifactImporter(settings, registry, load_smoke=lambda path: None)

    entry = importer.import_result(result, bundle)
    repeated = importer.import_result(result, bundle)

    assert repeated == entry
    assert entry.status == AdapterStatus.CANDIDATE
    assert Path(entry.path).is_relative_to(settings.qlora.output_dir.resolve())
    assert entry.training_job_id == "job-1"
    assert entry.training_node_id == "training-01"
    assert entry.submitted_by_node_id == "inference-01"
    assert entry.imported_by_node_id == settings.deployment.node.id
    assert entry.base_model_revision == settings.model.revision
    assert entry.adapter_hash is not None
    assert len(registry.list()) == 1
    importer.validate_entry(entry)
    (Path(entry.path) / "adapter_config.json").write_text("{}", "utf-8")
    with pytest.raises(ValueError, match="hash"):
        importer.validate_entry(entry)


def test_load_failure_records_attempt_without_registry_change(tmp_path: Path) -> None:
    settings, bundle, result = _artifacts(tmp_path)
    registry = AdapterRegistry(settings)
    importer = CandidateArtifactImporter(
        settings,
        registry,
        load_smoke=lambda path: (_ for _ in ()).throw(RuntimeError("cannot load")),
    )

    with pytest.raises(RuntimeError, match="cannot load"):
        importer.import_result(result, bundle)

    assert registry.list() == []
    attempts = json.loads(importer.attempt_path.read_text("utf-8"))["attempts"]
    assert attempts[-1]["status"] == "failed"
    assert not list((settings.qlora.output_dir / "imported").glob("adapter-*"))


def test_registry_rejects_duplicate_hash_and_training_job(tmp_path: Path) -> None:
    settings, _bundle, _result = _artifacts(tmp_path)
    registry = AdapterRegistry(settings)
    values = {
        "adapter_path": tmp_path / "adapter",
        "dataset_path": tmp_path / "dataset",
        "dataset_hash": "dataset-hash",
        "adapter_hash": "adapter-hash",
        "training_job_id": "job-1",
    }
    registry.register_candidate(adapter_id="adapter-1", **values)

    with pytest.raises(ValueError, match="hash already"):
        registry.register_candidate(
            adapter_id="adapter-2", **{**values, "training_job_id": "job-2"}
        )
    with pytest.raises(ValueError, match="Training job already"):
        registry.register_candidate(
            adapter_id="adapter-3", **{**values, "adapter_hash": "other-hash"}
        )


def _artifacts(tmp_path: Path):
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "qlora": settings.qlora.model_copy(update={"output_dir": tmp_path / "adapters"}),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": tmp_path / "registry.json", "eval_sets": []}
            ),
        }
    )
    dataset = b'{"input":"one"}\n'
    bundle_manifest = TrainingBundleManifest(
        job_id="job-1", attempt_id="attempt-1", created_at=datetime.now(UTC),
        submitter_node_id="inference-01", submitter_hostname="inference",
        base_model_id=settings.model.primary_id, base_model_revision=settings.model.revision,
        processor_revision=settings.model.processor_revision,
        source_event_sequence_start=1, source_event_sequence_end=1,
        dataset_hash=sha256_bytes(dataset), dataset_record_count=1,
        evaluation_set_hash=sha256_bytes(b""), evaluation_record_count=0,
        chat_template_version="gemma-v1", dataset_format_version="dream-v2",
        qlora_hyperparameters={"r": 16},
    )
    bundle = TrainingArtifactContract().finalize_bundle(
        tmp_path / "bundles", bundle_manifest, dataset=dataset, evaluation_set=b""
    )
    adapter_files = {
        "adapter/adapter_config.json": b'{"peft_type":"LORA"}\n',
        "adapter/training_manifest.json": b"{}\n",
    }
    result_manifest = TrainingResultManifest(
        job_id="job-1", attempt_id="attempt-1", created_at=datetime.now(UTC),
        worker_node_id="training-01", worker_hostname="worker", status="succeeded",
        candidate_adapter_id="adapter-1",
        candidate_adapter_hash=sha256_file_map(adapter_files),
        base_model_id=settings.model.primary_id,
        base_model_revision=settings.model.revision,
    )
    result = TrainingArtifactContract().finalize_result(
        tmp_path / "results", result_manifest, training_metrics={}, evaluation={"score": 1.0},
        adapter_files=adapter_files,
    )
    return settings, bundle, result

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from kagya.training import (
    TrainingArtifactContract,
    TrainingBundleManifest,
    TrainingResultManifest,
    sha256_bytes,
    sha256_file_map,
)


def test_bundle_finalize_validate_and_provenance(tmp_path: Path) -> None:
    dataset = b'{"input":"one"}\n'
    evaluation = b'{"input":"eval"}\n'
    manifest = _bundle_manifest(dataset, evaluation)

    path = TrainingArtifactContract().finalize_bundle(
        tmp_path, manifest, dataset=dataset, evaluation_set=evaluation
    )
    restored = TrainingArtifactContract().validate_bundle(
        path,
        expected_model_id="google/gemma-4-12B-it",
        expected_model_revision="model-commit-123",
        expected_processor_revision="processor-commit-123",
    )

    assert path.name == "training-job-1"
    assert restored.submitter_node_id == "inference-01"
    assert restored.source_episode_ids == ["episode-1"]
    assert restored.source_decision_ids == ["decision-1"]
    assert set(item.name for item in path.iterdir()) == {
        "manifest.json",
        "dataset.jsonl",
        "evaluation_set.jsonl",
        "checksums.sha256",
    }
    assert "checksums.sha256" not in (path / "checksums.sha256").read_text()


def test_bundle_rejects_tamper_incomplete_unknown_schema_and_overwrite(
    tmp_path: Path,
) -> None:
    dataset = b"one\n"
    evaluation = b"eval\n"
    contract = TrainingArtifactContract()
    manifest = _bundle_manifest(dataset, evaluation)
    path = contract.finalize_bundle(
        tmp_path, manifest, dataset=dataset, evaluation_set=evaluation
    )

    with pytest.raises(FileExistsError):
        contract.finalize_bundle(
            tmp_path, manifest, dataset=dataset, evaluation_set=evaluation
        )
    (path / "dataset.jsonl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch|checksum mismatch"):
        contract.validate_bundle(path)

    incomplete = tmp_path / "training-incomplete"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="file set"):
        contract.validate_bundle(incomplete)

    raw = manifest.model_dump(mode="json")
    raw["schema_version"] = 2
    with pytest.raises(ValidationError):
        TrainingBundleManifest.model_validate(raw)


def test_bundle_rejects_symlink_and_revision_mismatch(tmp_path: Path) -> None:
    dataset = b"one\n"
    evaluation = b"eval\n"
    contract = TrainingArtifactContract()
    path = contract.finalize_bundle(
        tmp_path,
        _bundle_manifest(dataset, evaluation),
        dataset=dataset,
        evaluation_set=evaluation,
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        contract.validate_bundle(path, expected_model_revision="other")
    (path / "extra-link").symlink_to(path / "dataset.jsonl")
    with pytest.raises(ValueError, match="symlinks"):
        contract.validate_bundle(path)


def test_result_finalize_validate_and_reject_partial_or_unsafe_path(
    tmp_path: Path,
) -> None:
    contract = TrainingArtifactContract()
    manifest = _result_manifest()

    path = contract.finalize_result(
        tmp_path,
        manifest,
        training_metrics={"loss": 0.4},
        evaluation={"score": 0.8},
        adapter_files={"adapter/adapter_config.json": b"{}\n"},
    )
    restored = contract.validate_result(
        path,
        expected_job_id="job-1",
        expected_attempt_id="attempt-1",
        expected_model_id="google/gemma-4-12B-it",
        expected_model_revision="model-commit-123",
    )

    assert restored.worker_node_id == "training-01"
    assert restored.candidate_adapter_hash == sha256_file_map(
        {"adapter/adapter_config.json": b"{}\n"}
    )
    with pytest.raises(ValueError, match="unsafe|under adapter"):
        contract.finalize_result(
            tmp_path / "unsafe",
            manifest.model_copy(update={"job_id": "job-2"}),
            training_metrics={},
            evaluation={},
            adapter_files={"../escape": b"bad"},
        )
    with pytest.raises(ValueError, match="adapter files"):
        contract.finalize_result(
            tmp_path / "partial",
            manifest.model_copy(update={"job_id": "job-3"}),
            training_metrics={},
            evaluation={},
        )


def test_failed_result_has_no_adapter_but_keeps_failure_provenance(
    tmp_path: Path,
) -> None:
    manifest = _result_manifest().model_copy(
        update={
            "status": "failed",
            "candidate_adapter_id": None,
            "candidate_adapter_hash": None,
            "failure_category": "training_nan",
            "error": "non-finite loss",
        }
    )

    path = TrainingArtifactContract().finalize_result(
        tmp_path,
        manifest,
        training_metrics={},
        evaluation={},
    )
    restored = TrainingArtifactContract().validate_result(path)

    assert restored.status == "failed"
    assert restored.failure_category == "training_nan"


def _bundle_manifest(dataset: bytes, evaluation: bytes) -> TrainingBundleManifest:
    return TrainingBundleManifest(
        job_id="job-1",
        attempt_id="attempt-1",
        created_at=datetime.now(UTC),
        submitter_node_id="inference-01",
        submitter_hostname="inference-host",
        base_model_id="google/gemma-4-12B-it",
        base_model_revision="model-commit-123",
        processor_revision="processor-commit-123",
        source_event_sequence_start=10,
        source_event_sequence_end=20,
        source_episode_ids=["episode-1"],
        source_decision_ids=["decision-1"],
        dataset_hash=sha256_bytes(dataset),
        dataset_record_count=1,
        evaluation_set_hash=sha256_bytes(evaluation),
        evaluation_record_count=1,
        chat_template_version="gemma-v1",
        dataset_format_version="decision-v1",
        qlora_hyperparameters={"r": 16, "learning_rate": 0.0002},
        required_capabilities=["cuda", "bitsandbytes"],
    )


def _result_manifest() -> TrainingResultManifest:
    return TrainingResultManifest(
        job_id="job-1",
        attempt_id="attempt-1",
        created_at=datetime.now(UTC),
        worker_node_id="training-01",
        worker_hostname="training-host",
        status="succeeded",
        candidate_adapter_id="adapter-1",
        candidate_adapter_hash=sha256_file_map(
            {"adapter/adapter_config.json": b"{}\n"}
        ),
        base_model_id="google/gemma-4-12B-it",
        base_model_revision="model-commit-123",
    )

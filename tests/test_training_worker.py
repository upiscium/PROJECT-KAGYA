from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil

import pytest

from kagya.config import Settings, load_settings
from kagya.training import (
    TrainingArtifactContract,
    TrainingBundleManifest,
    TrainingWorkerService,
    WorkerJob,
    WorkerJobStatus,
    sha256_bytes,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_worker_validates_runs_and_persists_idempotent_job(tmp_path: Path) -> None:
    settings = _worker_settings(tmp_path)
    staging = _staged_bundle(settings, "job-1")
    service = TrainingWorkerService(settings)

    job, created = service.submit(
        staging, settings.deployment.training.worker.result_directory, "request-1"
    )
    completed = service.execute(job.job_id)

    assert created is True
    assert completed.status == WorkerJobStatus.SUCCEEDED
    assert Path(completed.result_path).is_dir()
    assert (
        TrainingWorkerService(settings).inspect(job.job_id).status
        == WorkerJobStatus.SUCCEEDED
    )

    duplicate_staging = _staged_bundle(settings, "job-1")
    duplicate, duplicate_created = service.submit(
        duplicate_staging,
        settings.deployment.training.worker.result_directory,
        "request-1",
    )
    assert duplicate_created is False
    assert duplicate.job_id == job.job_id


def test_worker_rejects_partial_transfer_and_unapproved_submitter(
    tmp_path: Path,
) -> None:
    settings = _worker_settings(tmp_path)
    worker = settings.deployment.training.worker
    partial = worker.inbox_directory / ".training-partial.tmp"
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="file set"):
        TrainingWorkerService(settings).submit(
            partial, worker.result_directory, "partial"
        )

    staging = _staged_bundle(settings, "unapproved", submitter="other-node")
    with pytest.raises(ValueError, match="not allowed"):
        TrainingWorkerService(settings).submit(
            staging, worker.result_directory, "unapproved"
        )


def test_worker_enforces_concurrency_and_cancel_is_durable(tmp_path: Path) -> None:
    settings = _worker_settings(tmp_path)
    worker = settings.deployment.training.worker
    service = TrainingWorkerService(settings)
    first, _ = service.submit(
        _staged_bundle(settings, "job-1"), worker.result_directory, "request-1"
    )

    with pytest.raises(RuntimeError, match="max_concurrent_jobs"):
        service.submit(
            _staged_bundle(settings, "job-2"), worker.result_directory, "request-2"
        )

    cancelled = service.cancel(first.job_id)
    assert cancelled.status == WorkerJobStatus.CANCELLED
    assert (
        TrainingWorkerService(settings).inspect(first.job_id).status
        == WorkerJobStatus.CANCELLED
    )


def test_worker_rejects_job_id_reuse_with_different_idempotency_key(
    tmp_path: Path,
) -> None:
    settings = _worker_settings(tmp_path)
    worker = settings.deployment.training.worker
    service = TrainingWorkerService(settings)
    service.submit(
        _staged_bundle(settings, "job-1"), worker.result_directory, "request-1"
    )

    with pytest.raises(ValueError, match="already assigned"):
        service.submit(
            _staged_bundle(settings, "job-1"), worker.result_directory, "request-2"
        )


def test_worker_health_reports_node_revisions_and_capacity(tmp_path: Path) -> None:
    settings = _worker_settings(tmp_path)

    health = TrainingWorkerService(settings).health()

    assert health["node_id"] == "training-01"
    assert health["hostname"]
    assert health["heartbeat"]
    assert health["model_id"] == settings.model.primary_id
    assert health["model_revision"] == settings.model.revision
    assert health["processor_revision"] == settings.model.processor_revision
    assert health["active_jobs"] == 0
    assert "gpu" in health
    assert "driver" in health["gpu"] or health["gpu"]["available"] is False


def test_worker_cleanup_honors_retention_and_failed_job_policy(tmp_path: Path) -> None:
    settings = _worker_settings(tmp_path)
    service = TrainingWorkerService(settings)
    old = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    paths = [tmp_path / name for name in ("bundle", "work", "result")]
    for path in paths:
        path.mkdir()
    job = WorkerJob(
        job_id="old-job",
        attempt_id="old-attempt",
        idempotency_key="old-request",
        status=WorkerJobStatus.SUCCEEDED,
        bundle_path=str(paths[0]),
        work_path=str(paths[1]),
        result_path=str(paths[2]),
        submitter_node_id="inference-01",
        base_model_id=settings.model.primary_id,
        base_model_revision=settings.model.revision,
        processor_revision=settings.model.processor_revision,
        created_at=old,
        updated_at=old,
    )
    service.store.create(job, settings.deployment.training.worker.max_concurrent_jobs)
    failed_paths = [tmp_path / f"failed-{name}" for name in ("bundle", "work", "result")]
    for path in failed_paths:
        path.mkdir()
    service.store.create(
        WorkerJob(
            **{
                **job.__dict__,
                "job_id": "failed-job",
                "attempt_id": "failed-attempt",
                "idempotency_key": "failed-request",
                "status": WorkerJobStatus.FAILED,
                "bundle_path": str(failed_paths[0]),
                "work_path": str(failed_paths[1]),
                "result_path": str(failed_paths[2]),
            }
        ),
        settings.deployment.training.worker.max_concurrent_jobs,
    )

    result = service.cleanup(30)

    assert set(result["removed"]) == {str(path) for path in paths}
    assert not any(path.exists() for path in paths)
    assert all(path.exists() for path in failed_paths)


def _worker_settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    raw = settings.model_dump(mode="python")
    raw["model"]["revision"] = "model-commit-123"
    raw["model"]["processor_revision"] = "processor-commit-123"
    raw["qlora"]["dry_run"] = True
    raw["qlora"]["output_dir"] = tmp_path / "trainer-output"
    raw["deployment"] = {
        "mode": "split",
        "node": {"id": "training-01", "role": "training_worker"},
        "training": {
            "backend": "worker",
            "worker": {
                "inbox_directory": tmp_path / "inbox",
                "work_directory": tmp_path / "work",
                "result_directory": tmp_path / "results",
                "max_concurrent_jobs": 1,
                "retain_failed_jobs": True,
                "allowed_submitters": ["inference-01"],
            },
        },
    }
    return Settings.model_validate(raw)


def _staged_bundle(
    settings: Settings, job_id: str, *, submitter: str = "inference-01"
) -> Path:
    worker = settings.deployment.training.worker
    source_root = worker.inbox_directory.parent / f"source-{job_id}"
    shutil.rmtree(source_root, ignore_errors=True)
    dataset = b'{"input":"hello","thought":"","output":"world"}\n'
    evaluation = b""
    manifest = TrainingBundleManifest(
        job_id=job_id,
        attempt_id=f"attempt-{job_id}",
        created_at=datetime.now(UTC),
        submitter_node_id=submitter,
        submitter_hostname="inference-host",
        base_model_id=settings.model.primary_id,
        base_model_revision=settings.model.revision,
        processor_revision=settings.model.processor_revision,
        source_event_sequence_start=1,
        source_event_sequence_end=1,
        dataset_hash=sha256_bytes(dataset),
        dataset_record_count=1,
        evaluation_set_hash=sha256_bytes(evaluation),
        evaluation_record_count=0,
        chat_template_version="gemma-v1",
        dataset_format_version="dream-v2",
        qlora_hyperparameters={"r": 8},
    )
    source = TrainingArtifactContract().finalize_bundle(
        source_root, manifest, dataset=dataset, evaluation_set=evaluation
    )
    worker.inbox_directory.mkdir(parents=True, exist_ok=True)
    staging = worker.inbox_directory / f".training-{job_id}.tmp"
    shutil.copytree(source, staging)
    return staging

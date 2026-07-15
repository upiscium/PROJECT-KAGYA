from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from kagya.config.schema import ExpectedWorkerModelSettings, RemoteWorkerSettings
from kagya.training import (
    SSHTrainingBackend,
    TrainingArtifactContract,
    TrainingBundleManifest,
    TrainingJob,
    TrainingJobStatus,
    TrainingResultManifest,
    sha256_bytes,
    sha256_file_map,
)


def test_ssh_backend_uses_strict_argv_and_recovers_transient_disconnect(
    tmp_path: Path,
) -> None:
    bundle, job = _bundle_and_job(tmp_path)
    remote_result = _result(tmp_path, job)
    calls: list[list[str]] = []
    statuses = iter(["running", "succeeded"])
    disconnected = True

    def run(argv, **kwargs):
        nonlocal disconnected
        calls.append(argv)
        if argv[0] == "rsync":
            if "remote:/worker/results" in argv[-2]:
                _copy_contents(remote_result, Path(argv[-1]))
            return subprocess.CompletedProcess(argv, 0, "", "")
        action = argv[-3] if argv[-2] == "--job-id" else argv[-7]
        if action == "run":
            return _completed(argv, {"job_id": job.job_id, "status": "ready"})
        if action == "status" and disconnected:
            disconnected = False
            raise subprocess.CalledProcessError(255, argv, stderr="disconnected")
        if action == "status":
            return _completed(argv, {"job_id": job.job_id, "status": next(statuses)})
        raise AssertionError(argv)

    backend = SSHTrainingBackend(
        _remote_settings(tmp_path),
        tmp_path / "local-results",
        run=run,
        sleep=lambda _: None,
    )

    assert backend.submit(job, bundle) == job.job_id
    result = backend.fetch_result(job.job_id)

    assert result is not None
    assert result.adapter_id == "adapter-1"
    assert result.adapter_path.is_dir()
    ssh_call = next(call for call in calls if call[0] == "ssh")
    assert "StrictHostKeyChecking=yes" in ssh_call
    assert "UserKnownHostsFile=" + str(tmp_path / "known_hosts") in ssh_call
    assert all(not isinstance(argument, bytes) for call in calls for argument in call)


def test_ssh_backend_retries_duplicate_submit_without_new_remote_job(
    tmp_path: Path,
) -> None:
    bundle, job = _bundle_and_job(tmp_path)
    run_count = 0

    def run(argv, **kwargs):
        nonlocal run_count
        if argv[0] == "rsync":
            return subprocess.CompletedProcess(argv, 0, "", "")
        run_count += 1
        return _completed(argv, {"job_id": job.job_id, "status": "ready"})

    backend = SSHTrainingBackend(
        _remote_settings(tmp_path), tmp_path / "results", run=run
    )
    assert backend.submit(job, bundle) == job.job_id
    assert backend.submit(job, bundle) == job.job_id
    assert run_count == 2


def test_ssh_backend_timeout_cancels_remote_job(tmp_path: Path) -> None:
    bundle, job = _bundle_and_job(tmp_path)
    actions: list[str] = []
    clock = iter([0.0, 2.0])

    def run(argv, **kwargs):
        if argv[0] == "rsync":
            return subprocess.CompletedProcess(argv, 0, "", "")
        action = argv[-3] if argv[-2] == "--job-id" else argv[-7]
        actions.append(action)
        if action == "run":
            return _completed(argv, {"job_id": job.job_id, "status": "ready"})
        if action == "cancel":
            return _completed(argv, {"job_id": job.job_id, "status": "cancelled"})
        return _completed(argv, {"job_id": job.job_id, "status": "running"})

    settings = _remote_settings(tmp_path).model_copy(
        update={"job_timeout_seconds": 1.0}
    )
    backend = SSHTrainingBackend(
        settings,
        tmp_path / "results",
        run=run,
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )
    backend.submit(job, bundle)

    with pytest.raises(TimeoutError, match="timed out"):
        backend.fetch_result(job.job_id)
    assert "cancel" in actions


def test_ssh_backend_rejects_partial_download(tmp_path: Path) -> None:
    bundle, job = _bundle_and_job(tmp_path)

    def run(argv, **kwargs):
        if argv[0] == "rsync":
            if "remote:/worker/results" in argv[-2]:
                (Path(argv[-1]) / "result.json").write_text("{}")
            return subprocess.CompletedProcess(argv, 0, "", "")
        action = argv[-3] if argv[-2] == "--job-id" else argv[-7]
        status = "succeeded" if action == "status" else "ready"
        return _completed(argv, {"job_id": job.job_id, "status": status})

    backend = SSHTrainingBackend(
        _remote_settings(tmp_path), tmp_path / "results", run=run
    )
    backend.submit(job, bundle)

    with pytest.raises(ValueError, match="incomplete"):
        backend.fetch_result(job.job_id)
    assert not (tmp_path / "results" / "remote-results" / "result-job-1").exists()


def _remote_settings(tmp_path: Path) -> RemoteWorkerSettings:
    return RemoteWorkerSettings(
        node_id="training-01",
        host="remote",
        user="worker",
        identity_file=tmp_path / "identity",
        known_hosts_file=tmp_path / "known_hosts",
        remote_inbox="/worker/inbox",
        remote_results="/worker/results",
        command="/worker/bin/kagya-worker",
        connect_timeout_seconds=1.0,
        job_timeout_seconds=10.0,
        poll_interval_seconds=0.01,
        expected_worker_model=ExpectedWorkerModelSettings(
            model_id="model", revision="revision", processor_revision="processor"
        ),
    )


def _bundle_and_job(tmp_path: Path) -> tuple[Path, TrainingJob]:
    dataset = b'{"input":"hello"}\n'
    manifest = TrainingBundleManifest(
        job_id="job-1",
        attempt_id="attempt-1",
        created_at=datetime.now(UTC),
        submitter_node_id="inference-01",
        submitter_hostname="inference",
        base_model_id="model",
        base_model_revision="revision",
        processor_revision="processor",
        source_event_sequence_start=1,
        source_event_sequence_end=1,
        dataset_hash=sha256_bytes(dataset),
        dataset_record_count=1,
        evaluation_set_hash=sha256_bytes(b""),
        evaluation_record_count=0,
        chat_template_version="v1",
        dataset_format_version="v1",
        qlora_hyperparameters={"r": 8},
    )
    bundle = TrainingArtifactContract().finalize_bundle(
        tmp_path / "bundles", manifest, dataset=dataset, evaluation_set=b""
    )
    now = datetime.now(UTC).isoformat()
    job = TrainingJob(
        job_id="job-1",
        attempt_id="attempt-1",
        idempotency_key="request-1",
        status=TrainingJobStatus.READY,
        bundle_path=str(bundle),
        bundle_hash="hash",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        source_event_sequence_start=1,
        source_event_sequence_end=1,
        backend="ssh",
        remote_job_id=None,
        candidate_adapter_id=None,
        selected_episode_ids=(),
        semantic_memory_ids=(),
        created_at=now,
        updated_at=now,
    )
    return bundle, job


def _result(tmp_path: Path, job: TrainingJob) -> Path:
    adapter_files = {"adapter/adapter_config.json": b"{}\n"}
    return TrainingArtifactContract().finalize_result(
        tmp_path / "worker-results",
        TrainingResultManifest(
            job_id=job.job_id,
            attempt_id=job.attempt_id,
            created_at=datetime.now(UTC),
            worker_node_id="training-01",
            worker_hostname="worker",
            status="succeeded",
            candidate_adapter_id="adapter-1",
            candidate_adapter_hash=sha256_file_map(adapter_files),
            base_model_id=job.base_model_id,
            base_model_revision=job.base_model_revision,
        ),
        training_metrics={"dry_run": True, "training_records": 1},
        evaluation={},
        adapter_files=adapter_files,
    )


def _copy_contents(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _completed(argv: list[str], payload: dict[str, str]):
    return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

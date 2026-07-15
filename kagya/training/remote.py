"""SSH and rsync implementation of the remote training backend."""

from collections.abc import Callable
import json
from pathlib import Path
import shutil
import subprocess
from threading import Event
import time
from typing import Any

from kagya.config.schema import RemoteWorkerSettings
from kagya.learning import QloraTrainingResult
from kagya.training.artifacts import TrainingArtifactContract, sha256_bytes
from kagya.training.jobs import TrainingJob, TrainingJobStatus


class SSHTrainingBackend:
    def __init__(
        self,
        settings: RemoteWorkerSettings,
        local_result_root: Path,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.local_result_root = local_result_root
        self._run = run
        self._sleep = sleep
        self._monotonic = monotonic
        self._contract = TrainingArtifactContract()
        self._jobs: dict[str, TrainingJob] = {}
        self._shutdown = Event()

    def submit(self, job: TrainingJob, bundle_path: Path) -> str:
        _validate_identifier(job.job_id, "job ID")
        remote_idempotency_key = sha256_bytes(job.idempotency_key.encode("utf-8"))
        manifest = self._contract.validate_bundle(
            bundle_path,
            expected_model_id=self.settings.expected_worker_model.model_id,
            expected_model_revision=self.settings.expected_worker_model.revision,
            expected_processor_revision=self.settings.expected_worker_model.processor_revision,
            expected_parent_adapter_id=job.parent_adapter_id,
        )
        if manifest.job_id != job.job_id or manifest.attempt_id != job.attempt_id:
            raise ValueError("training job does not match bundle provenance")
        remote_staging = self.settings.remote_inbox / f".training-{job.job_id}.tmp"
        self._run_checked(
            self._rsync_base()
            + [f"{bundle_path}/", f"{self._target()}:{remote_staging}/"]
        )
        response = self._run_checked(
            self._ssh_base()
            + [
                str(self.settings.command),
                "run",
                "--bundle",
                str(remote_staging),
                "--output",
                str(self.settings.remote_results),
                "--idempotency-key",
                remote_idempotency_key,
            ]
        )
        payload = _json_output(response.stdout)
        remote_job_id = str(payload.get("job_id", ""))
        _validate_identifier(remote_job_id, "remote job ID")
        if remote_job_id != job.job_id:
            raise RuntimeError("remote worker returned a mismatched job ID")
        self._jobs[job.job_id] = job
        return remote_job_id

    def inspect(self, job_id: str) -> TrainingJobStatus:
        payload = self._worker_command("status", job_id)
        status = str(payload.get("status", ""))
        mapping = {
            "ready": TrainingJobStatus.DISPATCHED,
            "running": TrainingJobStatus.RUNNING,
            "succeeded": TrainingJobStatus.SUCCEEDED,
            "failed": TrainingJobStatus.FAILED,
            "cancelled": TrainingJobStatus.CANCELLED,
        }
        try:
            return mapping[status]
        except KeyError as exc:
            raise RuntimeError(
                f"remote worker returned unknown status: {status}"
            ) from exc

    def cancel(self, job_id: str) -> bool:
        status = str(self._worker_command("cancel", job_id).get("status", ""))
        return status == "cancelled"

    def fetch_result(self, job_id: str) -> QloraTrainingResult | None:
        _validate_identifier(job_id, "job ID")
        deadline = self._monotonic() + self.settings.job_timeout_seconds
        while True:
            if self._shutdown.is_set():
                raise InterruptedError("remote backend polling stopped")
            status = self.inspect(job_id)
            if status == TrainingJobStatus.SUCCEEDED:
                break
            if status == TrainingJobStatus.CANCELLED:
                return None
            if status == TrainingJobStatus.FAILED:
                raise RuntimeError("remote training job failed")
            if self._monotonic() >= deadline:
                self.cancel(job_id)
                raise TimeoutError(f"remote training job timed out: {job_id}")
            if self._shutdown.wait(self.settings.poll_interval_seconds):
                raise InterruptedError("remote backend polling stopped")

        job = self._jobs.get(job_id)
        if job is None:
            raise RuntimeError("remote training job provenance is unavailable")
        destination_root = self.local_result_root / "remote-results"
        destination_root.mkdir(parents=True, exist_ok=True)
        final_path = destination_root / f"result-{job_id}"
        if not final_path.exists():
            staging = destination_root / f".download-result-{job_id}"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            try:
                self._run_checked(
                    self._rsync_base()
                    + [
                        f"{self._target()}:{self.settings.remote_results}/result-{job_id}/",
                        f"{staging}/",
                    ]
                )
                self._contract.validate_result(
                    staging,
                    expected_job_id=job.job_id,
                    expected_attempt_id=job.attempt_id,
                    expected_model_id=job.base_model_id,
                    expected_model_revision=job.base_model_revision,
                    expected_parent_adapter_id=job.parent_adapter_id,
                )
                staging.rename(final_path)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        manifest = self._contract.validate_result(
            final_path,
            expected_job_id=job.job_id,
            expected_attempt_id=job.attempt_id,
            expected_model_id=job.base_model_id,
            expected_model_revision=job.base_model_revision,
            expected_parent_adapter_id=job.parent_adapter_id,
        )
        if manifest.status != "succeeded" or manifest.candidate_adapter_id is None:
            raise RuntimeError("remote worker did not publish a successful result")
        metrics = json.loads(
            (final_path / manifest.training_metrics_path).read_text("utf-8")
        )
        bundle_path = Path(job.bundle_path or "")
        bundle = self._contract.validate_bundle(bundle_path)
        return QloraTrainingResult(
            adapter_id=manifest.candidate_adapter_id,
            adapter_path=final_path / "adapter",
            dataset_path=bundle_path / bundle.dataset_path,
            dataset_hash=bundle.dataset_hash,
            dry_run=bool(metrics.get("dry_run", False)),
            training_records=int(
                metrics.get("training_records", bundle.dataset_record_count)
            ),
        )

    def attach(self, job: TrainingJob) -> None:
        _validate_identifier(job.job_id, "job ID")
        self._jobs[job.job_id] = job

    def shutdown(self) -> None:
        self._shutdown.set()

    def _worker_command(self, action: str, job_id: str) -> dict[str, Any]:
        _validate_identifier(job_id, "job ID")
        response = self._run_checked(
            self._ssh_base() + [str(self.settings.command), action, "--job-id", job_id]
        )
        return _json_output(response.stdout)

    def _run_checked(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(3):
            try:
                return self._run(
                    argv,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.job_timeout_seconds,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_error = (
                    exc if isinstance(exc, subprocess.CalledProcessError) else None
                )
                if attempt < 2:
                    self._sleep(min(self.settings.poll_interval_seconds, 1.0))
        if last_error is not None:
            raise RuntimeError(
                f"remote command failed: {last_error.stderr}"
            ) from last_error
        raise TimeoutError("remote command timed out")

    def _ssh_base(self) -> list[str]:
        return [
            "ssh",
            "-p",
            str(self.settings.port),
            "-i",
            str(self.settings.identity_file),
            "-o",
            f"UserKnownHostsFile={self.settings.known_hosts_file}",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(self.settings.connect_timeout_seconds))}",
            "--",
            self._target(),
        ]

    def _rsync_base(self) -> list[str]:
        ssh = " ".join(
            [
                "ssh",
                "-p",
                str(self.settings.port),
                "-i",
                str(self.settings.identity_file),
                "-o",
                f"UserKnownHostsFile={self.settings.known_hosts_file}",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "BatchMode=yes",
            ]
        )
        return ["rsync", "--archive", "--protect-args", "--partial", "-e", ssh]

    def _target(self) -> str:
        return f"{self.settings.user}@{self.settings.host}"


def _json_output(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("remote worker returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("remote worker returned invalid payload")
    return payload


def _validate_identifier(value: str, label: str) -> None:
    if not value or any(
        not (character.isalnum() or character in "._-") for character in value
    ):
        raise ValueError(f"{label} contains unsafe characters")

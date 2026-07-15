"""Durable filesystem runtime for training-worker jobs."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
from typing import Any, Iterator
from uuid import uuid4

from kagya.config import Settings
from kagya.learning import QloraTrainer
from kagya.training.artifacts import (
    TrainingArtifactContract,
    TrainingResultManifest,
    sha256_file_map,
)


class WorkerJobStatus(StrEnum):
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


WORKER_TERMINAL_STATUSES = {
    WorkerJobStatus.SUCCEEDED,
    WorkerJobStatus.FAILED,
    WorkerJobStatus.CANCELLED,
}


@dataclass(frozen=True)
class WorkerJob:
    job_id: str
    attempt_id: str
    idempotency_key: str
    status: WorkerJobStatus
    bundle_path: str
    work_path: str
    result_path: str
    submitter_node_id: str
    base_model_id: str
    base_model_revision: str
    processor_revision: str
    created_at: str
    updated_at: str
    pid: int | None = None
    error: str | None = None
    schema_version: int = 1

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "WorkerJob":
        data = dict(value)
        data["status"] = WorkerJobStatus(data["status"])
        return cls(**data)


class WorkerJobStore:
    """Process-safe JSON store shared by worker CLI invocations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    def create(
        self, job: WorkerJob, max_concurrent_jobs: int
    ) -> tuple[WorkerJob, bool]:
        with self._locked_jobs() as jobs:
            for job_id, current in tuple(jobs.items()):
                if (
                    current.status == WorkerJobStatus.RUNNING
                    and current.pid is not None
                    and not _pid_exists(current.pid)
                ):
                    jobs[job_id] = replace(
                        current,
                        status=WorkerJobStatus.FAILED,
                        updated_at=_now(),
                        pid=None,
                        error="worker process exited before completion",
                    )
            same_job = jobs.get(job.job_id)
            if same_job is not None:
                if (
                    same_job.attempt_id != job.attempt_id
                    or same_job.idempotency_key != job.idempotency_key
                ):
                    raise ValueError(
                        "worker job ID is already assigned to another request"
                    )
                return same_job, False
            existing = next(
                (
                    item
                    for item in jobs.values()
                    if item.idempotency_key == job.idempotency_key
                ),
                None,
            )
            if existing is not None:
                if (
                    existing.job_id != job.job_id
                    or existing.attempt_id != job.attempt_id
                ):
                    raise ValueError(
                        "idempotency key is already assigned to another job"
                    )
                return existing, False
            active = sum(
                item.status in {WorkerJobStatus.READY, WorkerJobStatus.RUNNING}
                for item in jobs.values()
            )
            if active >= max_concurrent_jobs:
                raise RuntimeError("worker has reached max_concurrent_jobs")
            jobs[job.job_id] = job
            return job, True

    def get(self, job_id: str) -> WorkerJob:
        _validate_identifier(job_id, "job ID")
        with self._locked_jobs() as jobs:
            try:
                return jobs[job_id]
            except KeyError as exc:
                raise ValueError(f"Unknown worker job: {job_id}") from exc

    def update(self, job_id: str, **changes: Any) -> WorkerJob:
        _validate_identifier(job_id, "job ID")
        with self._locked_jobs() as jobs:
            try:
                current = jobs[job_id]
            except KeyError as exc:
                raise ValueError(f"Unknown worker job: {job_id}") from exc
            updated = replace(current, updated_at=_now(), **changes)
            jobs[job_id] = updated
            return updated

    def claim(self, job_id: str, pid: int) -> WorkerJob | None:
        _validate_identifier(job_id, "job ID")
        with self._locked_jobs() as jobs:
            try:
                current = jobs[job_id]
            except KeyError as exc:
                raise ValueError(f"Unknown worker job: {job_id}") from exc
            if current.status != WorkerJobStatus.READY:
                return None
            claimed = replace(
                current,
                status=WorkerJobStatus.RUNNING,
                updated_at=_now(),
                pid=pid,
                error=None,
            )
            jobs[job_id] = claimed
            return claimed

    @contextmanager
    def _locked_jobs(self) -> Iterator[dict[str, WorkerJob]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            jobs = self._load()
            try:
                yield jobs
                self._save(jobs)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[str, WorkerJob]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text("utf-8"))
        return {
            job.job_id: job
            for value in raw.get("jobs", [])
            for job in [WorkerJob.from_json(value)]
        }

    def _save(self, jobs: dict[str, WorkerJob]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid4()}.tmp")
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(
                {
                    "schema_version": 1,
                    "jobs": [
                        asdict(job)
                        for job in sorted(
                            jobs.values(), key=lambda item: item.created_at
                        )
                    ],
                },
                output,
                ensure_ascii=False,
                sort_keys=True,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)


class TrainingWorkerService:
    def __init__(self, settings: Settings) -> None:
        worker = settings.deployment.training.worker
        if worker is None:
            raise RuntimeError("Training worker settings are required")
        self.settings = settings
        self.worker = worker
        self.contract = TrainingArtifactContract()
        self.store = WorkerJobStore(worker.work_directory / "worker_jobs.json")

    def submit(
        self, bundle_path: Path, output_path: Path, idempotency_key: str | None
    ) -> tuple[WorkerJob, bool]:
        inbox = self.worker.inbox_directory.resolve()
        staging = bundle_path.resolve()
        if (
            staging.parent != inbox
            or not staging.name.startswith(".")
            or not staging.name.endswith(".tmp")
        ):
            raise ValueError(
                "bundle staging path must be a direct temporary child of worker inbox"
            )
        if output_path.resolve() != self.worker.result_directory.resolve():
            raise ValueError(
                "output path must match configured worker result directory"
            )

        final_bundle = inbox / staging.name[1:-4]
        if final_bundle.exists():
            manifest = self.contract.validate_bundle(
                final_bundle,
                expected_model_id=self.settings.model.primary_id,
                expected_model_revision=self.settings.model.revision,
                expected_processor_revision=self.settings.model.processor_revision,
            )
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.rename(staging, final_bundle)
            try:
                manifest = self.contract.validate_bundle(
                    final_bundle,
                    expected_model_id=self.settings.model.primary_id,
                    expected_model_revision=self.settings.model.revision,
                    expected_processor_revision=self.settings.model.processor_revision,
                )
            except Exception:
                os.rename(final_bundle, staging)
                raise
        if manifest.submitter_node_id not in self.worker.allowed_submitters:
            raise ValueError("bundle submitter node ID is not allowed")
        if manifest.job_id != final_bundle.name.removeprefix("training-"):
            raise ValueError("bundle directory does not match manifest job ID")
        idempotency_key = idempotency_key or manifest.attempt_id
        _validate_identifier(idempotency_key, "idempotency key")

        now = _now()
        job = WorkerJob(
            job_id=manifest.job_id,
            attempt_id=manifest.attempt_id,
            idempotency_key=idempotency_key,
            status=WorkerJobStatus.READY,
            bundle_path=str(final_bundle),
            work_path=str(self.worker.work_directory / f"training-{manifest.job_id}"),
            result_path=str(output_path / f"result-{manifest.job_id}"),
            submitter_node_id=manifest.submitter_node_id,
            base_model_id=manifest.base_model_id,
            base_model_revision=manifest.base_model_revision,
            processor_revision=manifest.processor_revision,
            created_at=now,
            updated_at=now,
        )
        return self.store.create(job, self.worker.max_concurrent_jobs)

    def execute(self, job_id: str) -> WorkerJob:
        job = self.store.get(job_id)
        if job.status in WORKER_TERMINAL_STATUSES:
            return job
        if job.status == WorkerJobStatus.RUNNING and job.pid != os.getpid():
            return job
        bundle_path = Path(job.bundle_path)
        work_path = Path(job.work_path)
        manifest = self.contract.validate_bundle(
            bundle_path,
            expected_model_id=self.settings.model.primary_id,
            expected_model_revision=self.settings.model.revision,
            expected_processor_revision=self.settings.model.processor_revision,
        )
        if manifest.submitter_node_id not in self.worker.allowed_submitters:
            raise ValueError("bundle submitter node ID is not allowed")
        if self.store.claim(job_id, os.getpid()) is None:
            return self.store.get(job_id)
        try:
            if work_path.exists():
                shutil.rmtree(work_path)
            shutil.copytree(bundle_path, work_path)
        except Exception as exc:
            self._finalize_failure(job, manifest, str(exc))
            return self.store.update(
                job_id,
                status=WorkerJobStatus.FAILED,
                error=str(exc),
                pid=None,
            )
        try:
            result = QloraTrainer(self.settings).train(
                work_path / manifest.dataset_path
            )
            adapter_files = {
                (
                    Path("adapter") / item.relative_to(result.adapter_path)
                ).as_posix(): item.read_bytes()
                for item in result.adapter_path.rglob("*")
                if item.is_file() and not item.is_symlink()
            }
            result_manifest = TrainingResultManifest(
                job_id=job.job_id,
                attempt_id=job.attempt_id,
                created_at=datetime.now(UTC),
                worker_node_id=self.settings.deployment.node.id,
                worker_hostname=os.uname().nodename,
                status="succeeded",
                candidate_adapter_id=result.adapter_id,
                candidate_adapter_hash=sha256_file_map(adapter_files),
                base_model_id=manifest.base_model_id,
                base_model_revision=manifest.base_model_revision,
                parent_adapter_id=manifest.parent_adapter_id,
            )
            result_path = self.contract.finalize_result(
                self.worker.result_directory,
                result_manifest,
                training_metrics={
                    "dry_run": result.dry_run,
                    "training_records": result.training_records,
                    "dataset_hash": result.dataset_hash,
                },
                evaluation={},
                adapter_files=adapter_files,
            )
            return self.store.update(
                job_id,
                status=WorkerJobStatus.SUCCEEDED,
                result_path=str(result_path),
                pid=None,
            )
        except Exception as exc:
            current = self.store.get(job_id)
            if current.status == WorkerJobStatus.CANCELLED:
                return current
            self._finalize_failure(job, manifest, str(exc))
            return self.store.update(
                job_id,
                status=WorkerJobStatus.FAILED,
                error=str(exc),
                pid=None,
            )

    def inspect(self, job_id: str) -> WorkerJob:
        job = self.store.get(job_id)
        if (
            job.status == WorkerJobStatus.RUNNING
            and job.pid is not None
            and not _pid_exists(job.pid)
        ):
            return self.store.update(
                job_id,
                status=WorkerJobStatus.FAILED,
                pid=None,
                error="worker process exited before completion",
            )
        return job

    def cancel(self, job_id: str) -> WorkerJob:
        job = self.store.get(job_id)
        if job.status in WORKER_TERMINAL_STATUSES:
            return job
        cancelled = self.store.update(
            job_id, status=WorkerJobStatus.CANCELLED, pid=None, error="cancelled"
        )
        if job.pid is not None and job.pid != os.getpid():
            try:
                os.kill(job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return cancelled

    def _finalize_failure(self, job: WorkerJob, manifest, error: str) -> None:
        result_path = Path(job.result_path)
        if result_path.exists():
            return
        self.contract.finalize_result(
            self.worker.result_directory,
            TrainingResultManifest(
                job_id=job.job_id,
                attempt_id=job.attempt_id,
                created_at=datetime.now(UTC),
                worker_node_id=self.settings.deployment.node.id,
                worker_hostname=os.uname().nodename,
                status="failed",
                base_model_id=manifest.base_model_id,
                base_model_revision=manifest.base_model_revision,
                parent_adapter_id=manifest.parent_adapter_id,
                failure_category="training_error",
                error=error,
            ),
            training_metrics={},
            evaluation={},
        )


def _validate_identifier(value: str, label: str) -> None:
    if not value or any(
        not (character.isalnum() or character in "._-") for character in value
    ):
        raise ValueError(f"{label} contains unsafe characters")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

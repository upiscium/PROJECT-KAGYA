from dataclasses import dataclass
from pathlib import Path
from threading import Event

from kagya.config import load_settings
from kagya.training import (
    ConsolidationPreparation,
    SleepCoordinator,
    TrainingJobRegistry,
    TrainingJobStatus,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_job_registry_is_persistent_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    registry = TrainingJobRegistry(path)

    first, created = registry.create(
        idempotency_key="attempt-1",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="local",
        job_id="job-1",
    )
    duplicate, duplicate_created = registry.create(
        idempotency_key="attempt-1",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="local",
    )
    registry.update("job-1", status=TrainingJobStatus.RUNNING)

    restored = TrainingJobRegistry(path)

    assert created is True
    assert duplicate_created is False
    assert duplicate.job_id == first.job_id
    assert restored.get("job-1").status == TrainingJobStatus.RUNNING


def test_coordinator_cancels_prepared_job_without_publishing(tmp_path: Path) -> None:
    entered = Event()
    release = Event()
    consolidator = _FakeConsolidator()
    builder = _BlockingBuilder(tmp_path, entered, release)
    coordinator = _coordinator(tmp_path, consolidator, builder, _FakeBackend())

    job = coordinator.create_job("cancel-me")
    assert entered.wait(1)
    coordinator.cancel(job.job_id)
    release.set()
    coordinator.shutdown()

    restored = TrainingJobRegistry(tmp_path / "jobs.json").get(job.job_id)
    assert restored.status == TrainingJobStatus.CANCELLED
    assert consolidator.completed is False
    assert consolidator.failed is True


def test_backend_failure_is_persisted_and_staged_memory_is_not_published(
    tmp_path: Path,
) -> None:
    consolidator = _FakeConsolidator()
    coordinator = _coordinator(
        tmp_path, consolidator, _ImmediateBuilder(tmp_path), _FailingBackend()
    )

    job = coordinator.create_job("fail-me")
    coordinator.shutdown()

    restored = TrainingJobRegistry(tmp_path / "jobs.json").get(job.job_id)
    assert restored.status == TrainingJobStatus.FAILED
    assert "backend failed" in (restored.error or "")
    assert consolidator.completed is False
    assert consolidator.failed is True

    retry = coordinator.retry(job.job_id)
    coordinator.shutdown()
    assert retry.retry_count == 1
    assert coordinator.inspect(retry.job_id).status == TrainingJobStatus.FAILED


def test_coordinator_marks_interrupted_local_job_failed_on_restart(
    tmp_path: Path,
) -> None:
    registry = TrainingJobRegistry(tmp_path / "jobs.json")
    job, _ = registry.create(
        idempotency_key="interrupted",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="local",
    )
    registry.update(job.job_id, status=TrainingJobStatus.RUNNING)

    coordinator = _coordinator(
        tmp_path,
        _FakeConsolidator(),
        _ImmediateBuilder(tmp_path),
        _FakeBackend(),
    )

    recovered = coordinator.inspect(job.job_id)
    assert recovered.status == TrainingJobStatus.FAILED
    assert "restarted" in (recovered.error or "")


@dataclass(frozen=True)
class _Episode:
    id: str = "episode-1"
    processing_sequence: int = 3


class _FakeConsolidator:
    def __init__(self) -> None:
        self.completed = False
        self.failed = False

    def prepare(self, attempt_id: str) -> ConsolidationPreparation:
        return ConsolidationPreparation((_Episode(),), ("semantic-1",))

    def complete(self, preparation, attempt_id: str) -> None:
        self.completed = True

    def fail(self, preparation, attempt_id: str) -> None:
        self.failed = True


class _BlockingBuilder:
    def __init__(self, root: Path, entered: Event, release: Event) -> None:
        self.root = root
        self.entered = entered
        self.release = release

    def build(self, job, episodes) -> Path:
        self.entered.set()
        self.release.wait(1)
        path = self.root / "bundle"
        path.mkdir(exist_ok=True)
        (path / "checksums.sha256").write_bytes(b"hashes")
        return path


class _ImmediateBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root

    def build(self, job, episodes) -> Path:
        path = self.root / "bundle"
        path.mkdir(exist_ok=True)
        (path / "checksums.sha256").write_bytes(b"hashes")
        return path


class _FakeBackend:
    def submit(self, job, bundle_path: Path) -> str:
        return job.job_id

    def inspect(self, job_id: str):
        return TrainingJobStatus.SUCCEEDED

    def cancel(self, job_id: str) -> bool:
        return True

    def fetch_result(self, job_id: str):
        return None


class _FailingBackend(_FakeBackend):
    def submit(self, job, bundle_path: Path) -> str:
        raise RuntimeError("backend failed")


class _AdapterRegistry:
    def register_candidate(self, **kwargs):
        raise AssertionError("candidate must not be registered")


def _coordinator(tmp_path, consolidator, builder, backend) -> SleepCoordinator:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "sleep": settings.sleep.model_copy(
                update={
                    "job_registry_path": tmp_path / "jobs.json",
                    "training_artifact_directory": tmp_path / "artifacts",
                }
            )
        }
    )
    return SleepCoordinator(
        settings,
        consolidator,
        builder,
        TrainingJobRegistry(tmp_path / "jobs.json"),
        backend,
        _AdapterRegistry(),
    )

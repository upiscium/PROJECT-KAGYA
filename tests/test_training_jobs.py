from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from threading import Event

from kagya.api.routes.training import _adapter_projection, _job_projection
from kagya.config import load_settings
from kagya.config.schema import (
    ExpectedWorkerModelSettings,
    RemoteWorkerSettings,
    TrainingBackendType,
)
from kagya.learning import QloraTrainingResult
from kagya.learning.adapter_registry import AdapterRegistry
from kagya.training import (
    ConsolidationPreparation,
    LocalTrainingBackend,
    SleepCoordinator,
    TrainingJobRegistry,
    TrainingJobStatus,
)
from kagya.training.jobs import _artifact_digest


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


def test_job_registry_persists_lifecycle_timestamps_and_migrates_legacy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    registry = TrainingJobRegistry(path)
    job, _ = registry.create(
        idempotency_key="timestamps",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="local",
        job_id="job-timestamps",
    )
    assert job.started_at is None
    assert job.completed_at is None

    running = registry.transition(job.job_id, TrainingJobStatus.RUNNING)
    assert running.started_at is not None
    assert running.completed_at is None
    succeeded = registry.transition(job.job_id, TrainingJobStatus.SUCCEEDED)
    assert succeeded.started_at == running.started_at
    completed = registry.transition(job.job_id, TrainingJobStatus.COMPLETED)
    assert completed.started_at == running.started_at
    assert completed.completed_at is not None
    assert registry.update(job.job_id, stale=True).completed_at == completed.completed_at

    for status in (TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED):
        terminal, _ = registry.create(
            idempotency_key=f"terminal-{status.value}",
            base_model_id="model",
            base_model_revision="revision",
            parent_adapter_id=None,
            backend="local",
            job_id=f"job-{status.value}",
        )
        registry.transition(terminal.job_id, TrainingJobStatus.RUNNING)
        terminal = registry.transition(terminal.job_id, status)
        assert terminal.started_at is not None
        assert terminal.completed_at is None

    legacy = asdict(completed)
    legacy.pop("started_at")
    legacy.pop("completed_at")
    legacy["schema_version"] = 2
    legacy["phase_started_at"] = "2026-01-02T00:00:00+00:00"
    path.write_text(json.dumps({"schema_version": 1, "jobs": [legacy]}))

    migrated = TrainingJobRegistry(path).get(job.job_id)
    assert migrated.started_at is None
    assert migrated.completed_at == "2026-01-02T00:00:00+00:00"
    assert migrated.schema_version == 3


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
    duplicate_retry = coordinator.retry(job.job_id)
    assert duplicate_retry.job_id == retry.job_id


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


def test_coordinator_resumes_persisted_remote_job(tmp_path: Path) -> None:
    registry = TrainingJobRegistry(tmp_path / "jobs.json")
    job, _ = registry.create(
        idempotency_key="remote-request",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="ssh",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    registry.update(
        job.job_id,
        status=TrainingJobStatus.RUNNING,
        bundle_path=str(bundle),
        remote_job_id=job.job_id,
    )
    consolidator = _RecoveringConsolidator()
    backend = _RecoveringBackend(tmp_path)
    coordinator = SleepCoordinator(
        load_settings(CONFIG_PATH),
        consolidator,
        _ImmediateBuilder(tmp_path),
        TrainingJobRegistry(tmp_path / "jobs.json"),
        backend,
        _RegisteringAdapterRegistry(),
    )

    coordinator.shutdown()

    restored = TrainingJobRegistry(tmp_path / "jobs.json").get(job.job_id)
    assert backend.attached == job.job_id
    assert restored.status == TrainingJobStatus.COMPLETED
    assert restored.candidate_adapter_id == "adapter-remote"
    assert consolidator.completed is True


def test_reconcile_distinguishes_unreachable_worker_without_losing_job(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(
        tmp_path,
        _FakeConsolidator(),
        _ImmediateBuilder(tmp_path),
        _UnreachableBackend(),
    )
    job, _ = coordinator.registry.create(
        idempotency_key="remote",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="ssh",
    )
    coordinator.registry.update(
        job.job_id,
        status=TrainingJobStatus.RUNNING,
        remote_last_contact=(datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
    )

    reconciled = coordinator.reconcile(job.job_id)

    assert reconciled.status == TrainingJobStatus.RUNNING
    assert reconciled.failure_category == "worker_unreachable"
    assert reconciled.retryable is True
    assert reconciled.stale is True


def test_reconcile_reports_orphan_result_and_cleanup_removes_only_old_artifacts(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(
        tmp_path,
        _FakeConsolidator(),
        _ImmediateBuilder(tmp_path),
        _OrphanBackend(),
    )
    job, _ = coordinator.registry.create(
        idempotency_key="cleanup",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="local",
    )
    bundle = tmp_path / "old-bundle"
    bundle.mkdir()
    old = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(bundle, (old, old))
    coordinator.registry.update(
        job.job_id,
        status=TrainingJobStatus.COMPLETED,
        bundle_path=str(bundle),
    )
    orphan = tmp_path / "artifacts" / "remote-results" / "result-orphan"
    orphan.mkdir(parents=True)

    reconciled = coordinator.reconcile_all()
    cleaned = coordinator.cleanup(now=datetime.now(UTC))

    assert reconciled["orphan_result_job_ids"] == ["orphan"]
    assert reconciled["orphan_remote_job_ids"] == ["remote-orphan"]
    assert str(bundle) in cleaned["removed"]
    assert not bundle.exists()
    assert orphan.exists()


def test_standalone_finalize_persists_job_adapter_provenance(tmp_path: Path) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "deployment": settings.deployment.model_copy(
                update={"node": settings.deployment.node.model_copy(update={"id": "standalone-node"})}
            ),
            "sleep": settings.sleep.model_copy(
                update={"job_registry_path": tmp_path / "jobs.json"}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": tmp_path / "adapters.json"}
            ),
        }
    )
    registry = AdapterRegistry(settings)
    backend = LocalTrainingBackend(_SuccessfulTrainer(settings, tmp_path))
    coordinator = SleepCoordinator(
        settings,
        _FakeConsolidator(),
        _ImmediateBuilder(tmp_path),
        TrainingJobRegistry(tmp_path / "jobs.json"),
        backend,
        registry,
    )

    job = coordinator.create_job("local-request")
    coordinator.shutdown()

    restored = TrainingJobRegistry(tmp_path / "jobs.json").get(job.job_id)
    assert restored.status == TrainingJobStatus.COMPLETED
    assert restored.worker_node_id == "standalone-node"
    assert restored.candidate_adapter_id == "adapter-local"
    assert restored.result_digest is None
    adapter = registry.lookup("adapter-local")
    assert adapter is not None
    assert adapter.training_job_id == job.job_id
    assert adapter.training_node_id == "standalone-node"
    assert adapter.submitted_by_node_id == "standalone-node"
    assert adapter.imported_by_node_id == "standalone-node"

    job_projection = _job_projection(restored, {adapter.adapter_id: adapter})
    adapter_projection = _adapter_projection(
        adapter,
        {restored.job_id: restored},
        [],
        [],
        set(),
    )
    assert job_projection.candidate_adapter_id == adapter.adapter_id
    assert adapter_projection.training_job_id == restored.job_id


def test_running_local_job_has_cockpit_worker_linkage(tmp_path: Path) -> None:
    entered = Event()
    release = Event()
    settings = _settings(tmp_path)
    backend = LocalTrainingBackend(
        _BlockingTrainer(settings, tmp_path, entered, release)
    )
    coordinator = SleepCoordinator(
        settings,
        _FakeConsolidator(),
        _ImmediateBuilder(tmp_path),
        TrainingJobRegistry(tmp_path / "jobs.json"),
        backend,
        _RegisteringAdapterRegistry(),
    )

    job = coordinator.create_job("running-local")
    assert entered.wait(1)
    running = coordinator.inspect(job.job_id)

    assert running.status == TrainingJobStatus.RUNNING
    assert running.worker_node_id == settings.deployment.node.id
    assert _job_projection(running, {}).worker_node_id == settings.deployment.node.id

    release.set()
    coordinator.shutdown()


def test_running_remote_job_has_configured_worker_linkage(tmp_path: Path) -> None:
    entered = Event()
    release = Event()
    settings = _settings(tmp_path)
    remote = RemoteWorkerSettings(
        node_id="training-node",
        host="worker.example",
        user="worker",
        identity_file=tmp_path / "identity",
        known_hosts_file=tmp_path / "known-hosts",
        remote_inbox="/worker/inbox",
        remote_results="/worker/results",
        command="/worker/bin/kagya-worker",
        expected_worker_model=ExpectedWorkerModelSettings(
            model_id=settings.model.primary_id,
            revision=settings.model.revision,
            processor_revision=settings.model.processor_revision,
        ),
    )
    settings = settings.model_copy(
        update={
            "deployment": settings.deployment.model_copy(
                update={
                    "training": settings.deployment.training.model_copy(
                        update={
                            "backend": TrainingBackendType.SSH,
                            "remote_worker": remote,
                        }
                    )
                }
            )
        }
    )
    backend = _BlockingSubmitBackend(entered, release)
    coordinator = SleepCoordinator(
        settings,
        _FakeConsolidator(),
        _ImmediateBuilder(tmp_path),
        TrainingJobRegistry(tmp_path / "jobs.json"),
        backend,
        _AdapterRegistry(),
    )

    job = coordinator.create_job("running-remote")
    assert entered.wait(1)
    running = coordinator.inspect(job.job_id)

    assert running.status == TrainingJobStatus.RUNNING
    assert running.worker_node_id == "training-node"
    assert _job_projection(running, {}).worker_node_id == "training-node"

    release.set()
    coordinator.shutdown()


def test_remote_finalize_persists_validated_result_artifact_digest(
    tmp_path: Path,
) -> None:
    registry = TrainingJobRegistry(tmp_path / "jobs.json")
    importer = _ResultImporter()
    coordinator = SleepCoordinator(
        load_settings(CONFIG_PATH),
        _FakeConsolidator(),
        _ImmediateBuilder(tmp_path),
        registry,
        _FakeBackend(),
        _AdapterRegistry(),
        candidate_importer=importer,
    )
    job, _ = registry.create(
        idempotency_key="remote-result",
        base_model_id="model",
        base_model_revision="revision",
        parent_adapter_id=None,
        backend="ssh",
        job_id="job-remote",
    )
    artifact = tmp_path / "result-job-remote"
    artifact.mkdir()
    (artifact / "result.json").write_text('{"status":"succeeded"}\n')
    adapter_path = artifact / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter.bin").write_bytes(b"adapter-bytes")
    dataset = tmp_path / "bundle" / "dataset.jsonl"
    dataset.parent.mkdir()
    dataset.write_text("{}\n")
    result = QloraTrainingResult(
        adapter_id="adapter-remote",
        adapter_path=adapter_path,
        dataset_path=dataset,
        dataset_hash="dataset-hash",
        dry_run=False,
        training_records=1,
        artifact_path=artifact,
    )

    entry = coordinator._finalize_subject_state(
        ConsolidationPreparation((), ()),
        job,
        result,
    )

    digest = registry.get(job.job_id).result_digest
    assert entry.adapter_id == "adapter-remote"
    assert importer.result_path == artifact
    assert digest == _artifact_digest(artifact)
    assert digest is not None
    assert len(digest) == 64
    assert digest.islower()


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


class _Memory:
    def get_episodic(self, episode_id: str):
        return None


class _RecoveringConsolidator(_FakeConsolidator):
    def __init__(self) -> None:
        super().__init__()
        self.memory = _Memory()


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

    def job_metadata(self, job_id: str):
        return {}


class _BlockingSubmitBackend(_FakeBackend):
    def __init__(self, entered: Event, release: Event) -> None:
        self.entered = entered
        self.release = release

    def submit(self, job, bundle_path: Path) -> str:
        self.entered.set()
        self.release.wait(1)
        return job.job_id


class _FailingBackend(_FakeBackend):
    def submit(self, job, bundle_path: Path) -> str:
        raise RuntimeError("backend failed")


class _UnreachableBackend(_FakeBackend):
    def attach(self, job) -> None:
        return None

    def inspect(self, job_id: str):
        raise RuntimeError("remote command failed: network partition")


class _OrphanBackend(_FakeBackend):
    def node_status(self):
        return {
            "reachable": True,
            "jobs": [{"job_id": "remote-orphan", "status": "succeeded"}],
        }


class _RecoveringBackend(_FakeBackend):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.attached = None

    def attach(self, job) -> None:
        self.attached = job.job_id

    def fetch_result(self, job_id: str):
        adapter = self.root / "adapter-remote"
        adapter.mkdir(exist_ok=True)
        dataset = self.root / "dataset.jsonl"
        dataset.write_text("{}\n")
        return QloraTrainingResult(
            adapter_id="adapter-remote",
            adapter_path=adapter,
            dataset_path=dataset,
            dataset_hash="hash",
            dry_run=True,
            training_records=1,
        )


class _SuccessfulTrainer:
    def __init__(self, settings, root: Path) -> None:
        self.settings = settings
        self.root = root

    def train_bundle(self, bundle_path: Path):
        adapter = self.root / "adapter-local"
        adapter.mkdir(exist_ok=True)
        dataset = self.root / "dataset.jsonl"
        dataset.write_text('{}\n')
        return QloraTrainingResult(
            adapter_id="adapter-local",
            adapter_path=adapter,
            dataset_path=dataset,
            dataset_hash="hash",
            dry_run=True,
            training_records=1,
            adapter_hash="a" * 64,
        )


class _BlockingTrainer(_SuccessfulTrainer):
    def __init__(
        self,
        settings,
        root: Path,
        entered: Event,
        release: Event,
    ) -> None:
        super().__init__(settings, root)
        self.entered = entered
        self.release = release

    def train_bundle(self, bundle_path: Path):
        self.entered.set()
        self.release.wait(1)
        return super().train_bundle(bundle_path)


class _AdapterRegistry:
    def lookup(self, adapter_id: str):
        return None

    def register_candidate(self, **kwargs):
        raise AssertionError("candidate must not be registered")


class _ResultImporter:
    result_path: Path | None = None

    def import_result(self, result_path: Path, bundle_path: Path):
        self.result_path = result_path
        return _AdapterEntry("adapter-remote")


@dataclass(frozen=True)
class _AdapterEntry:
    adapter_id: str


class _RegisteringAdapterRegistry:
    def lookup(self, adapter_id: str):
        return None

    def register_candidate(self, **kwargs):
        return _AdapterEntry(kwargs["adapter_id"])


def _coordinator(tmp_path, consolidator, builder, backend) -> SleepCoordinator:
    settings = _settings(tmp_path)
    return SleepCoordinator(
        settings,
        consolidator,
        builder,
        TrainingJobRegistry(tmp_path / "jobs.json"),
        backend,
        _AdapterRegistry(),
    )


def _settings(tmp_path: Path):
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "sleep": settings.sleep.model_copy(
                update={
                    "job_registry_path": tmp_path / "jobs.json",
                    "training_artifact_directory": tmp_path / "artifacts",
                }
            )
        }
    )

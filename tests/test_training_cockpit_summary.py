from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from kagya.api.dependencies import (
    get_adapter_registry,
    get_adapter_runtime_manager,
    get_agent_runtime,
    get_sleep_coordinator,
)
from kagya.api.server import create_app
from kagya.config import Settings, load_settings
from kagya.config.schema import (
    DeploymentMode,
    ExpectedWorkerModelSettings,
    NodeRole,
    RemoteWorkerSettings,
    TrainingBackendType,
)
from kagya.learning.adapter_registry import AdapterEntry, AdapterStatus
from kagya.learning.adapter_runtime import AdapterActivationRecord
from kagya.runtime import AgentEventType
from kagya.training.jobs import TrainingJob, TrainingJobStatus


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def test_cockpit_training_is_available_without_auth(tmp_path: Path, monkeypatch) -> None:
    client, _runtime, _coordinator = _client(tmp_path, monkeypatch)

    assert _get(client).status_code == 200


def test_cockpit_training_standalone_node_and_read_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    client, runtime, coordinator = _client(tmp_path, monkeypatch)
    before = dict(client.app.state._state)

    response = _get(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"][0]["node_id"] == "node-main"
    assert payload["nodes"][0]["role"] == "inference"
    assert payload["nodes"][0]["backend"] == "local"
    assert payload["nodes"][0]["status"] == "online"
    assert runtime.events == [AgentEventType.TRAINING_READ]
    assert coordinator.node_status_calls == 0
    assert dict(client.app.state._state) == before


def test_cockpit_training_split_uses_configured_inventory_and_cached_health(
    tmp_path: Path, monkeypatch
) -> None:
    health = [
        {"node_id": "training-01", "reachable": True, "model_id": "wrong", "raw": "PRIVATE_SENTINEL"},
        {"node_id": "training-01", "reachable": True, "model_id": "wrong", "raw": "PRIVATE_SENTINEL"},
        {"node_id": "bad node", "reachable": True, "model_id": "wrong", "raw": "PRIVATE_SENTINEL"},
    ]
    client, _runtime, _coordinator = _client(tmp_path, monkeypatch, split=True, health=health)

    payload = _get(client).json()

    assert payload["node_count"] == 2
    assert payload["online_node_count"] == 1
    assert [(item["node_id"], item["role"], item["backend"], item["status"]) for item in payload["nodes"]] == [
        ("node-main", "inference", "ssh", "online"),
        ("training-01", "worker", "ssh", "unavailable"),
    ]
    assert "PRIVATE_SENTINEL" not in json.dumps(payload)


def test_cockpit_training_counts_limit_duplicates_and_strict_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    jobs = [_job(f"job-{index}", status=TrainingJobStatus.RUNNING, candidate_adapter_id=f"adapter-{index}") for index in range(5)]
    jobs.append(_job("job-dup", status=TrainingJobStatus.FAILED))
    jobs.append(_job("job-dup", status=TrainingJobStatus.FAILED))
    adapters = [
        _adapter("adapter-0", training_job_id="job-0", training_node_id="worker-1"),
        _adapter("adapter-1", training_job_id=None, training_node_id="worker-1"),
        _adapter("adapter-dup"),
        _adapter("adapter-dup"),
    ]
    client, _runtime, _coordinator = _client(tmp_path, monkeypatch, jobs=jobs, adapters=adapters)

    payload = _get(client, "?limit=1").json()

    assert payload["running_job_count"] == 5
    assert payload["failed_job_count"] == 0
    assert len(payload["jobs"]) == 1
    assert all(item["job_id"] != "unavailable" for item in payload["jobs"])
    assert len(payload["adapters"]) == 1
    assert payload["adapters"][0]["adapter_id"] != "adapter-dup"
    assert payload["adapters"][0]["training_job_id"] is None


def test_cockpit_training_activation_rollback_and_evaluation_binding(
    tmp_path: Path, monkeypatch
) -> None:
    adapter_a = _adapter("adapter-a", status=AdapterStatus.ACTIVE, adapter_hash=DIGEST_A)
    adapter_b = _adapter("adapter-b", status=AdapterStatus.APPROVED, adapter_hash=DIGEST_B)
    adapter_failed = _adapter("adapter-failed", gate=False)
    adapter_real = _adapter("adapter-real", real=True)
    adapter_corrupt = _adapter("adapter-corrupt", real=True, state="quarantined")
    history = [
        _activation("activate", "adapter-a", DIGEST_A, 1, "2026-01-01T00:00:00+00:00"),
        _activation("activate", "adapter-a", DIGEST_A, 3, "2026-01-01T00:00:03+00:00"),
        AdapterActivationRecord("rollback", "adapter-b", DIGEST_B, "adapter-a", DIGEST_A, 4, "2026-01-01T00:00:04+00:00"),
    ]
    journal = _Journal([
        _journal_record(3, "event-activate", "adapter:adapter-a"),
        _journal_record(4, "event-rollback", "adapter:adapter-a"),
    ])
    client, _runtime, _coordinator = _client(
        tmp_path,
        monkeypatch,
        adapters=[adapter_a, adapter_b, adapter_failed, adapter_real, adapter_corrupt],
        history=history,
        journal=journal,
    )

    adapters = {item["adapter_id"]: item for item in _get(client).json()["adapters"]}

    assert adapters["adapter-a"]["activation_event_id"] == "event-activate"
    assert adapters["adapter-a"]["activation_event_sequence"] == 3
    assert adapters["adapter-a"]["rollback_event_id"] == "event-rollback"
    assert adapters["adapter-b"]["rollback_event_id"] is None
    assert adapters["adapter-a"]["evaluation_status"] == "passed"
    assert adapters["adapter-failed"]["evaluation_status"] == "failed"
    assert adapters["adapter-real"]["evaluation_status"] == "passed"
    assert adapters["adapter-corrupt"]["evaluation_status"] == "corrupt"


def test_cockpit_training_marks_rollback_target_adapter_only(
    tmp_path: Path, monkeypatch
) -> None:
    target = _adapter("adapter-target", status=AdapterStatus.ARCHIVED)
    active = replace(
        _adapter("adapter-active", status=AdapterStatus.ACTIVE),
        rollback_target_id="adapter-target",
    )
    missing_target_active = replace(
        _adapter("adapter-active-missing", status=AdapterStatus.ACTIVE),
        rollback_target_id="missing-target",
    )
    archived_with_target = replace(
        _adapter("adapter-archived-source", status=AdapterStatus.ARCHIVED),
        rollback_target_id="adapter-unrelated",
    )
    unrelated = _adapter("adapter-unrelated", status=AdapterStatus.ARCHIVED)
    canary = replace(
        _adapter("adapter-canary"),
        rollout_state="canary",
    )
    client, _runtime, _coordinator = _client(
        tmp_path,
        monkeypatch,
        adapters=[
            target,
            active,
            missing_target_active,
            archived_with_target,
            unrelated,
            canary,
        ],
    )

    adapters = {item["adapter_id"]: item for item in _get(client).json()["adapters"]}

    assert adapters["adapter-target"]["rollback_candidate"] is True
    assert adapters["adapter-active"]["rollback_candidate"] is False
    assert adapters["adapter-active-missing"]["rollback_candidate"] is False
    assert adapters["adapter-unrelated"]["rollback_candidate"] is False
    assert adapters["adapter-canary"]["rollback_candidate"] is False


def test_cockpit_training_cross_record_mismatches_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    jobs = [_job("job-1", candidate_adapter_id="adapter-1", worker_node_id="worker-1")]
    mismatches = [
        _adapter("adapter-1", training_job_id="other-job", training_node_id="worker-1"),
        _adapter("adapter-2", training_job_id="job-1", training_node_id="worker-1"),
        _adapter("adapter-3", training_job_id="job-1", training_node_id="worker-1", base_model="other-model"),
        _adapter("adapter-4", training_job_id="job-1", training_node_id="worker-1", base_revision="other-rev"),
        _adapter("adapter-5", training_job_id="job-1", training_node_id="other-worker"),
    ]
    client, _runtime, _coordinator = _client(tmp_path, monkeypatch, jobs=jobs, adapters=mismatches)

    payload = _get(client).json()

    assert payload["jobs"][0]["candidate_adapter_id"] is None
    assert all(item["training_job_id"] is None for item in payload["adapters"])


def test_cockpit_training_uses_job_result_digest_not_adapter_hash(
    tmp_path: Path, monkeypatch
) -> None:
    result_digest = "c" * 64
    job = replace(
        _job("job-1", status=TrainingJobStatus.COMPLETED, candidate_adapter_id="adapter-1"),
        result_digest=result_digest,
    )
    adapter = _adapter("adapter-1", adapter_hash=DIGEST_A)
    client, _runtime, _coordinator = _client(
        tmp_path, monkeypatch, jobs=[job], adapters=[adapter]
    )

    payload = _get(client).json()

    assert payload["jobs"][0]["candidate_adapter_id"] == "adapter-1"
    assert payload["jobs"][0]["result_digest"] == result_digest
    assert payload["jobs"][0]["result_digest"] != adapter.adapter_hash


def _get(client: TestClient, suffix: str = ""):
    return client.get(f"/api/training/cockpit-summary{suffix}")


def _client(
    tmp_path: Path,
    monkeypatch,
    *,
    split: bool = False,
    health: list[dict] | None = None,
    jobs: list[TrainingJob] | None = None,
    adapters: list[AdapterEntry] | None = None,
    history: list[AdapterActivationRecord] | None = None,
    journal=None,
) -> tuple[TestClient, "_Runtime", "_Coordinator"]:
    settings = _settings(tmp_path, split=split)
    app = create_app(settings)
    runtime = _Runtime()
    coordinator = _Coordinator(health or [], jobs or [])
    app.dependency_overrides[get_agent_runtime] = lambda: runtime
    app.dependency_overrides[get_sleep_coordinator] = lambda: coordinator
    app.dependency_overrides[get_adapter_registry] = lambda: _Registry(adapters or [])
    app.dependency_overrides[get_adapter_runtime_manager] = lambda: _Manager(history or [])
    if journal is not None:
        app.state.event_journal = journal
    return TestClient(app), runtime, coordinator


class _Runtime:
    def __init__(self) -> None:
        self.events: list[AgentEventType] = []

    def execute(self, event_type, *, handler, **_kwargs):
        self.events.append(event_type)
        return SimpleNamespace(value=handler())


class _Coordinator:
    def __init__(self, health: list[dict], jobs: list[TrainingJob]) -> None:
        self.health = health
        self.jobs = jobs
        self.node_status_calls = 0

    def cached_node_status(self) -> list[dict]:
        return self.health

    def node_status(self) -> list[dict]:
        self.node_status_calls += 1
        raise AssertionError("cockpit summary must not probe node_status inside TRAINING_READ")

    def list_jobs(self) -> list[TrainingJob]:
        return self.jobs


class _Registry:
    def __init__(self, adapters: list[AdapterEntry]) -> None:
        self.adapters = adapters

    def list(self) -> list[AdapterEntry]:
        return self.adapters


class _Manager:
    def __init__(self, history: list[AdapterActivationRecord]) -> None:
        self._history = history

    def history(self) -> list[AdapterActivationRecord]:
        return self._history


class _Journal:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._records = records

    def recent(self, _limit: int):
        return self._records


def _settings(tmp_path: Path, *, split: bool) -> Settings:
    settings = load_settings(CONFIG_PATH)
    if split:
        settings = settings.model_copy(
            update={
                "model": settings.model.model_copy(
                    update={"revision": "a" * 40, "processor_revision": "b" * 40}
                )
            }
        )
    deployment = settings.deployment.model_copy(
        update={
            "mode": DeploymentMode.SPLIT if split else DeploymentMode.STANDALONE,
            "node": settings.deployment.node.model_copy(update={"id": "node-main", "role": NodeRole.INFERENCE if split else NodeRole.ALL}),
            "training": settings.deployment.training.model_copy(
                update={
                    "backend": TrainingBackendType.SSH if split else TrainingBackendType.LOCAL,
                    "remote_worker": _remote_worker(tmp_path, settings) if split else None,
                }
            ),
        }
    )
    return settings.model_copy(
        update={
            "deployment": deployment,
            "sleep": settings.sleep.model_copy(update={"job_registry_path": tmp_path / "jobs.json", "training_artifact_directory": tmp_path / "artifacts"}),
            "adapter_registry": settings.adapter_registry.model_copy(update={"path": tmp_path / "adapters.json", "eval_result_dir": tmp_path / "eval"}),
            "agent_state": settings.agent_state.model_copy(update={"path": tmp_path / "agent_state.json"}),
            "agent_journal": settings.agent_journal.model_copy(update={"path": tmp_path / "journal.jsonl"}),
            "agent_state_wal": settings.agent_state_wal.model_copy(update={"path": tmp_path / "wal.jsonl"}),
        }
    )


def _remote_worker(tmp_path: Path, settings: Settings) -> RemoteWorkerSettings:
    return RemoteWorkerSettings(
        node_id="training-01",
        host="training.example",
        port=22,
        user="worker",
        identity_file=tmp_path / "ssh_key_path_PRIVATE_SENTINEL",
        known_hosts_file=tmp_path / "known_hosts",
        remote_inbox=Path("/remote/inbox"),
        remote_results=Path("/remote/results"),
        command=Path("/usr/bin/worker-command"),
        expected_worker_model=ExpectedWorkerModelSettings(model_id=settings.model.primary_id, revision=settings.model.revision, processor_revision=settings.model.processor_revision),
    )


def _job(
    job_id: str,
    *,
    status: TrainingJobStatus = TrainingJobStatus.RUNNING,
    candidate_adapter_id: str | None = None,
    worker_node_id: str | None = "worker-1",
) -> TrainingJob:
    now = "2026-01-01T00:00:00+00:00"
    return TrainingJob(
        job_id=job_id,
        attempt_id=f"attempt-{job_id}",
        idempotency_key=f"idempotency-{job_id}-PRIVATE_SENTINEL",
        status=status,
        bundle_path="/private/path/PRIVATE_SENTINEL",
        bundle_hash=DIGEST_A,
        base_model_id="dummy-model",
        base_model_revision="dummy-revision",
        parent_adapter_id=None,
        source_event_sequence_start=1,
        source_event_sequence_end=2,
        backend="ssh",
        remote_job_id=job_id,
        candidate_adapter_id=candidate_adapter_id,
        selected_episode_ids=("episode-1", "episode-2"),
        semantic_memory_ids=(),
        created_at=now,
        updated_at=now,
        error="raw stderr PRIVATE_SENTINEL",
        worker_node_id=worker_node_id,
        failure_category=None,
    )


def _adapter(
    adapter_id: str,
    *,
    status: AdapterStatus = AdapterStatus.CANDIDATE,
    adapter_hash: str = DIGEST_A,
    training_job_id: str | None = "job-1",
    training_node_id: str | None = "worker-1",
    base_model: str = "dummy-model",
    base_revision: str = "dummy-revision",
    gate: bool = True,
    real: bool = False,
    state: str = "reconciled",
) -> AdapterEntry:
    entry = AdapterEntry(
        adapter_id=adapter_id,
        base_model=base_model,
        path="/private/path/PRIVATE_SENTINEL",
        status=status,
        dataset_path="dataset text PRIVATE_SENTINEL",
        dataset_hash=DIGEST_A,
        base_model_revision=base_revision,
        adapter_hash=adapter_hash,
        training_job_id=training_job_id,
        training_node_id=training_node_id,
        submitted_by_node_id="submitter-1",
        imported_by_node_id="importer-1",
    )
    if real:
        return replace(
            entry,
            real_model_behavioral_evaluation_id=f"eval-{adapter_id}",
            real_model_behavioral_candidate_adapter_hash=adapter_hash,
            real_model_behavioral_base_model_revision=base_revision,
            real_model_behavioral_gate_passed=gate,
            real_model_behavioral_artifact_state=state,
        )
    return replace(
        entry,
        behavioral_evaluation_id=f"eval-{adapter_id}",
        behavioral_candidate_adapter_hash=adapter_hash,
        behavioral_base_model_revision=base_revision,
        behavioral_gate_passed=gate,
        behavioral_artifact_state=state,
    )


def _activation(action: str, adapter_id: str, adapter_hash: str, sequence: int, created_at: str) -> AdapterActivationRecord:
    return AdapterActivationRecord(action, adapter_id, adapter_hash, None, None, sequence, created_at)


def _journal_record(sequence: int, event_id: str, target: str):
    return SimpleNamespace(
        processing_sequence=sequence,
        event_id=event_id,
        event_type=AgentEventType.ADAPTER_UPDATE.value,
        target=target,
        correlation_id=None,
        causation_id=None,
        timestamp=datetime.now(UTC),
    )

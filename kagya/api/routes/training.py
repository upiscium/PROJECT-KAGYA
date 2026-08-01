"""Distributed training node observability routes."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, TypeVar, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_adapter_registry,
    get_adapter_runtime_manager,
    get_agent_runtime,
    get_api_settings,
    get_dataset_governance,
    get_sleep_coordinator,
)
from kagya.config import DeploymentMode, Settings
from kagya.learning import AdapterRegistry, AdapterRuntimeManager
from kagya.learning.adapter_registry import AdapterEntry, AdapterStatus
from kagya.learning.adapter_runtime import AdapterActivationRecord
from kagya.runtime import AgentEventType, AgentRuntime, EventJournal, JournalRecord
from kagya.training import DatasetGovernanceStore, SleepCoordinator
from kagya.training.jobs import TrainingJob, TrainingJobStatus


RecordT = TypeVar("RecordT")
SafeId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]
SafeText = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9 ._+:/(),-]+$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BoundedCode = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")]


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CockpitTrainingNodeResponse(_ResponseModel):
    node_id: SafeId
    role: Literal["inference", "worker"]
    backend: SafeText
    status: Literal["online", "unavailable"]
    last_contact_at: datetime | None
    expected_model_id: SafeText | None
    expected_model_revision: SafeText | None
    expected_processor_revision: SafeText | None
    observed_model_id: SafeText | None
    observed_model_revision: SafeText | None
    model_matches_expected: bool | None
    gpu_name: SafeText | None
    cuda_version: SafeText | None
    driver_version: SafeText | None


class CockpitTrainingJobResponse(_ResponseModel):
    job_id: SafeId
    attempt_id: SafeId
    status: Literal[
        "preparing",
        "ready",
        "dispatched",
        "running",
        "succeeded",
        "importing",
        "completed",
        "failed",
        "cancelled",
        "unavailable",
    ]
    backend: SafeText | None
    created_at: datetime | None
    updated_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    source_event_start: int | None = Field(default=None, ge=0)
    source_event_end: int | None = Field(default=None, ge=0)
    selected_episode_count: int | None = Field(default=None, ge=0)
    remote_job_id: SafeId | None
    worker_node_id: SafeId | None
    retry_count: int | None = Field(default=None, ge=0)
    transferred_bytes: int | None = Field(default=None, ge=0)
    failure_code: BoundedCode | None
    candidate_adapter_id: SafeId | None
    import_status: Literal["not_started", "importing", "completed", "failed", "unavailable"]
    bundle_digest: Digest | None
    result_digest: Digest | None


class CockpitAdapterLineageResponse(_ResponseModel):
    adapter_id: SafeId
    status: SafeText
    adapter_hash: Digest | None
    base_model_id: SafeText | None
    base_model_revision: SafeText | None
    parent_adapter_id: SafeId | None
    training_job_id: SafeId | None
    training_node_id: SafeId | None
    submitted_by_node_id: SafeId | None
    imported_by_node_id: SafeId | None
    evaluation_id: SafeId | None
    evaluation_status: Literal["passed", "failed", "stale", "corrupt", "unavailable"]
    approved: bool
    active: bool
    rollback_candidate: bool
    activation_event_id: SafeId | None
    activation_event_sequence: int | None = Field(default=None, ge=1)
    rollback_event_id: SafeId | None
    rollback_event_sequence: int | None = Field(default=None, ge=1)


class CockpitTrainingSummaryResponse(_ResponseModel):
    node_count: int = Field(ge=0)
    online_node_count: int = Field(ge=0)
    running_job_count: int = Field(ge=0)
    failed_job_count: int = Field(ge=0)
    importing_job_count: int = Field(ge=0)
    active_adapter_count: int = Field(ge=0)
    candidate_adapter_count: int = Field(ge=0)
    nodes: list[CockpitTrainingNodeResponse]
    jobs: list[CockpitTrainingJobResponse]
    adapters: list[CockpitAdapterLineageResponse]


@dataclass(frozen=True)
class _ConfiguredNode:
    node_id: str
    role: Literal["inference", "worker"]
    backend: Literal["local", "ssh"]
    health: dict[str, Any]


router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/nodes")
def training_nodes(
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
) -> dict[str, list[dict]]:
    return {"nodes": coordinator.node_status()}


@router.get("/cockpit-summary", response_model=CockpitTrainingSummaryResponse)
def cockpit_training_summary(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    settings: Settings = Depends(get_api_settings),
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
    registry: AdapterRegistry = Depends(get_adapter_registry),
    manager: AdapterRuntimeManager = Depends(get_adapter_runtime_manager),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> CockpitTrainingSummaryResponse:
    raw_health = coordinator.cached_node_status()

    def build() -> CockpitTrainingSummaryResponse:
        configured_nodes = _configured_nodes(settings, raw_health)
        jobs = coordinator.list_jobs()
        adapters = registry.list()
        history = manager.history()
        journal = getattr(request.app.state, "event_journal", None)
        journal_records = _journal_records(journal)
        job_by_id = unique_by_id(jobs, lambda item: item.job_id)
        adapter_by_id = unique_by_id(adapters, lambda item: item.adapter_id)
        unique_jobs = list(job_by_id.values())
        unique_adapters = list(adapter_by_id.values())
        ordered_jobs = sorted(
            unique_jobs,
            key=lambda item: (item.updated_at, item.created_at, item.job_id),
            reverse=True,
        )
        ordered_adapters = sorted(
            unique_adapters,
            key=lambda item: (item.updated_at, item.created_at, item.adapter_id),
            reverse=True,
        )
        rollback_target_adapter_ids = {
            item.rollback_target_id
            for item in unique_adapters
            if item.status == AdapterStatus.ACTIVE
            and item.rollback_target_id is not None
        }
        nodes = [_node_projection(node, settings) for node in configured_nodes]
        return CockpitTrainingSummaryResponse(
            node_count=len(nodes),
            online_node_count=sum(node.status == "online" for node in nodes),
            running_job_count=sum(item.status == TrainingJobStatus.RUNNING for item in unique_jobs),
            failed_job_count=sum(item.status == TrainingJobStatus.FAILED for item in unique_jobs),
            importing_job_count=sum(item.status == TrainingJobStatus.IMPORTING for item in unique_jobs),
            active_adapter_count=sum(item.status.value == "active" for item in unique_adapters),
            candidate_adapter_count=sum(item.status.value == "candidate" for item in unique_adapters),
            nodes=nodes[:limit],
            jobs=[
                _job_projection(job, adapter_by_id)
                for job in ordered_jobs[:limit]
            ],
            adapters=[
                _adapter_projection(
                    adapter,
                    job_by_id,
                    history,
                    journal_records,
                    rollback_target_adapter_ids,
                )
                for adapter in ordered_adapters[:limit]
            ],
        )

    return execute_agent_event(
        runtime,
        AgentEventType.TRAINING_READ,
        source="api.training.cockpit_summary",
        handler=build,
        payload={"limit": limit},
    ).value


@router.get("/datasets")
def dataset_revisions(
    store: DatasetGovernanceStore = Depends(get_dataset_governance),
) -> dict[str, list[dict]]:
    return {"datasets": store.list_revisions()}


@router.get("/datasets/diff")
def dataset_revision_diff(
    from_revision: str = Query(alias="from"),
    to_revision: str = Query(alias="to"),
    store: DatasetGovernanceStore = Depends(get_dataset_governance),
) -> dict:
    try:
        return store.diff(from_revision, to_revision)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{revision}")
def dataset_revision(
    revision: str,
    store: DatasetGovernanceStore = Depends(get_dataset_governance),
) -> dict:
    try:
        dataset = store.get_revision(revision)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "manifest": {**dataset.manifest, "manifest_hash": dataset.manifest_hash},
        "records": [record.to_json() for record in dataset.records],
    }


def _configured_nodes(settings: Settings, health: list[dict[str, Any]]) -> list[_ConfiguredNode]:
    by_node_id = unique_by_id(
        (item for item in health if _safe_identifier(item.get("node_id")) is not None),
        lambda item: _safe_identifier(item.get("node_id")) or "",
    )
    if settings.deployment.mode == DeploymentMode.SPLIT and settings.deployment.training.remote_worker is not None:
        remote = settings.deployment.training.remote_worker
        return [
            _ConfiguredNode(settings.deployment.node.id, "inference", "ssh", {"reachable": True, "backend": "ssh"}),
            _ConfiguredNode(remote.node_id, "worker", "ssh", by_node_id.get(remote.node_id, {})),
        ]
    return [
        _ConfiguredNode(settings.deployment.node.id, "inference", "local", {"reachable": True, "backend": "local"})
    ]


def _node_projection(
    node: _ConfiguredNode, settings: Settings
) -> CockpitTrainingNodeResponse:
    reachable = bool(node.health.get("reachable", False))
    observed_model = _safe_optional_text(node.health.get("model_id")) if reachable else None
    observed_revision = _safe_optional_text(node.health.get("model_revision")) if reachable else None
    expected = settings.deployment.training.remote_worker.expected_worker_model if (
        node.role == "worker" and settings.deployment.training.remote_worker is not None
    ) else settings.model
    gpu = node.health.get("gpu") if isinstance(node.health.get("gpu"), dict) else {}
    devices = gpu.get("devices") if isinstance(gpu, dict) else None
    gpu_name = None
    if isinstance(devices, list) and devices:
        gpu_name = _safe_optional_text(devices[0])
    status: Literal["online", "unavailable"] = "online" if reachable else "unavailable"
    return CockpitTrainingNodeResponse(
        node_id=node.node_id,
        role=node.role,
        backend=node.backend,
        status=status,
        last_contact_at=_safe_datetime(node.health.get("last_contact") or node.health.get("heartbeat")),
        expected_model_id=_safe_optional_text(getattr(expected, "model_id", settings.model.primary_id)),
        expected_model_revision=_safe_optional_text(getattr(expected, "revision", settings.model.revision)),
        expected_processor_revision=_safe_optional_text(getattr(expected, "processor_revision", settings.model.processor_revision)),
        observed_model_id=observed_model,
        observed_model_revision=observed_revision,
        model_matches_expected=(
            None
            if not reachable or observed_model is None or observed_revision is None
            else observed_model == getattr(expected, "model_id", settings.model.primary_id)
            and observed_revision == getattr(expected, "revision", settings.model.revision)
        ),
        gpu_name=gpu_name,
        cuda_version=_safe_optional_text(gpu.get("cuda")) if isinstance(gpu, dict) else None,
        driver_version=_safe_optional_text(gpu.get("driver")) if isinstance(gpu, dict) else None,
    )


def _job_projection(
    job: TrainingJob | None,
    adapter_by_id: dict[str, AdapterEntry],
) -> CockpitTrainingJobResponse:
    if job is None:
        return CockpitTrainingJobResponse(
            job_id="unavailable",
            attempt_id="unavailable",
            status="unavailable",
            backend=None,
            created_at=None,
            updated_at=None,
            started_at=None,
            completed_at=None,
            source_event_start=None,
            source_event_end=None,
            selected_episode_count=None,
            remote_job_id=None,
            worker_node_id=None,
            retry_count=None,
            transferred_bytes=None,
            failure_code=None,
            candidate_adapter_id=None,
            import_status="unavailable",
            bundle_digest=None,
            result_digest=None,
        )
    adapter = adapter_by_id.get(job.candidate_adapter_id or "")
    if adapter is not None and not _job_adapter_matches(job, adapter):
        adapter = None
    return CockpitTrainingJobResponse(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        status=job.status.value,
        backend=_safe_optional_text(job.backend),
        created_at=_safe_datetime(job.created_at),
        updated_at=_safe_datetime(job.updated_at),
        started_at=_safe_datetime(job.started_at),
        completed_at=_safe_datetime(job.completed_at),
        source_event_start=job.source_event_sequence_start,
        source_event_end=job.source_event_sequence_end,
        selected_episode_count=len(job.selected_episode_ids),
        remote_job_id=_safe_identifier(job.remote_job_id),
        worker_node_id=_safe_identifier(job.worker_node_id),
        retry_count=job.retry_count,
        transferred_bytes=job.transferred_bytes,
        failure_code=_failure_code(job.failure_category),
        candidate_adapter_id=None if adapter is None else adapter.adapter_id,
        import_status=_import_status(job.import_status),
        bundle_digest=_digest(job.bundle_hash),
        result_digest=_digest(job.result_digest),
    )


def _adapter_projection(
    adapter: AdapterEntry | None,
    job_by_id: dict[str, TrainingJob],
    history: list[AdapterActivationRecord],
    journal_records: list[Any],
    rollback_target_adapter_ids: set[str],
) -> CockpitAdapterLineageResponse:
    if adapter is None:
        return CockpitAdapterLineageResponse(
            adapter_id="unavailable",
            status="unavailable",
            adapter_hash=None,
            base_model_id=None,
            base_model_revision=None,
            parent_adapter_id=None,
            training_job_id=None,
            training_node_id=None,
            submitted_by_node_id=None,
            imported_by_node_id=None,
            evaluation_id=None,
            evaluation_status="unavailable",
            approved=False,
            active=False,
            rollback_candidate=False,
            activation_event_id=None,
            activation_event_sequence=None,
            rollback_event_id=None,
            rollback_event_sequence=None,
        )
    job = job_by_id.get(adapter.training_job_id or "")
    if job is not None and not _job_adapter_matches(job, adapter):
        job = None
    activation = _latest_activation(adapter, history)
    rollback = _latest_rollback(adapter, history)
    evaluation_id, evaluation_status = _evaluation_binding(adapter)
    return CockpitAdapterLineageResponse(
        adapter_id=adapter.adapter_id,
        status=adapter.status.value,
        adapter_hash=_digest(adapter.adapter_hash),
        base_model_id=_safe_optional_text(adapter.base_model),
        base_model_revision=_safe_optional_text(adapter.base_model_revision),
        parent_adapter_id=_safe_identifier(adapter.parent_adapter_id),
        training_job_id=None if job is None else job.job_id,
        training_node_id=_safe_identifier(adapter.training_node_id),
        submitted_by_node_id=_safe_identifier(adapter.submitted_by_node_id),
        imported_by_node_id=_safe_identifier(adapter.imported_by_node_id),
        evaluation_id=_safe_identifier(evaluation_id),
        evaluation_status=evaluation_status,
        approved=adapter.status.value in {"approved", "active", "archived"},
        active=adapter.status.value == "active",
        rollback_candidate=adapter.adapter_id in rollback_target_adapter_ids,
        activation_event_id=_event_id(adapter, activation, journal_records),
        activation_event_sequence=None if activation is None else activation.activation_sequence,
        rollback_event_id=_event_id(adapter, rollback, journal_records),
        rollback_event_sequence=None if rollback is None else rollback.activation_sequence,
    )


def _job_adapter_matches(job: TrainingJob, adapter: AdapterEntry) -> bool:
    return (
        job.candidate_adapter_id == adapter.adapter_id
        and adapter.training_job_id == job.job_id
        and adapter.base_model == job.base_model_id
        and adapter.base_model_revision == job.base_model_revision
        and adapter.training_node_id is not None
        and job.worker_node_id is not None
        and adapter.training_node_id == job.worker_node_id
    )


def _latest_activation(
    adapter: AdapterEntry, records: list[AdapterActivationRecord]
) -> AdapterActivationRecord | None:
    matches = [
        record
        for record in records
        if record.action.startswith("activate")
        and record.adapter_id == adapter.adapter_id
        and record.adapter_hash == adapter.adapter_hash
    ]
    return None if not matches else max(matches, key=lambda item: (item.activation_sequence, item.created_at))


def _latest_rollback(
    adapter: AdapterEntry, records: list[AdapterActivationRecord]
) -> AdapterActivationRecord | None:
    matches = [
        record
        for record in records
        if record.action.startswith("rollback")
        and record.previous_adapter_id == adapter.adapter_id
        and record.previous_adapter_hash == adapter.adapter_hash
    ]
    return None if not matches else max(matches, key=lambda item: (item.activation_sequence, item.created_at))


def _event_id(
    adapter: AdapterEntry,
    record: AdapterActivationRecord | None,
    journal_records: list[JournalRecord],
) -> str | None:
    if record is None:
        return None
    matches = [
        item
        for item in journal_records
        if item.processing_sequence == record.activation_sequence
        and item.event_type == AgentEventType.ADAPTER_UPDATE.value
        and _journal_matches_adapter(item, adapter.adapter_id)
    ]
    if len(matches) != 1:
        return None
    return matches[0].event_id


def _journal_records(journal: EventJournal | None) -> list[JournalRecord]:
    if journal is None:
        return []
    return list(journal.recent(500))


def _journal_matches_adapter(record: JournalRecord, adapter_id: str) -> bool:
    refs = [record.target, record.correlation_id, record.causation_id]
    constrained = [ref for ref in refs if ref and "adapter" in ref]
    return not constrained or any(adapter_id in ref for ref in constrained)


def unique_by_id(
    values: Iterable[RecordT], identifier: Callable[[RecordT], str]
) -> dict[str, RecordT]:
    grouped: dict[str, list[RecordT]] = {}
    for item in values:
        grouped.setdefault(identifier(item), []).append(item)
    return {record_id: items[0] for record_id, items in grouped.items() if len(items) == 1}


def _safe_identifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if 1 <= len(text) <= 128 and all(c.isalnum() or c in "._:-" for c in text) else None


def _safe_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)[:160]
    return text if text and all(c.isalnum() or c in " ._+:/(),-" for c in text) else None


def _safe_optional_text(value: object) -> str | None:
    return _safe_text(value)


def _safe_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _digest(value: str | None) -> str | None:
    return value if value is not None and len(value) == 64 and all(c in "0123456789abcdef" for c in value) else None


def _import_status(value: str) -> Literal["not_started", "importing", "completed", "failed", "unavailable"]:
    if value in {"not_started", "importing", "completed", "failed"}:
        return cast(
            Literal["not_started", "importing", "completed", "failed", "unavailable"],
            value,
        )
    return "unavailable"


def _failure_code(value: str | None) -> str | None:
    mapping = {
        "worker_unreachable": "worker_unavailable",
        "worker_process_lost": "worker_unavailable",
        "remote_training_failed": "training_failed",
        "backend_failure": "training_failed",
        "training_failure": "training_failed",
        "training_error": "training_failed",
        "artifact_integrity": "result_invalid",
        "artifact_import": "import_failed",
        "cuda_oom": "cuda_oom",
        "non_finite_metrics": "non_finite_metrics",
        "timeout": "timeout",
        "cancelled": "cancelled",
    }
    if value is None:
        return None
    return mapping.get(value, "unknown_failure")


def _evaluation_binding(
    adapter: AdapterEntry,
) -> tuple[str | None, Literal["passed", "failed", "stale", "corrupt", "unavailable"]]:
    evaluation_id: str | None
    if adapter.real_model_behavioral_evaluation_id is not None:
        evaluation_id = adapter.real_model_behavioral_evaluation_id
        state = adapter.real_model_behavioral_artifact_state
        gate = adapter.real_model_behavioral_gate_passed
        candidate_hash = adapter.real_model_behavioral_candidate_adapter_hash
        base_revision = adapter.real_model_behavioral_base_model_revision
    else:
        evaluation_id = adapter.behavioral_evaluation_id
        state = adapter.behavioral_artifact_state
        gate = adapter.behavioral_gate_passed
        candidate_hash = adapter.behavioral_candidate_adapter_hash
        base_revision = adapter.behavioral_base_model_revision
    if evaluation_id is None:
        return None, "unavailable"
    if candidate_hash != adapter.adapter_hash or base_revision != adapter.base_model_revision:
        return None, "unavailable"
    if state in {"quarantined"}:
        return evaluation_id, "corrupt"
    if state in {"prepared", "finalized"}:
        return evaluation_id, "stale"
    if state == "reconciled":
        return evaluation_id, "passed" if gate is True else "failed"
    return None, "unavailable"

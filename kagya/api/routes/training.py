"""Distributed training node observability routes."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
from kagya.config import Settings
from kagya.learning import AdapterRegistry, AdapterRuntimeManager
from kagya.learning.adapter_registry import AdapterEntry
from kagya.learning.adapter_runtime import AdapterActivationRecord
from kagya.runtime import AgentEventType, AgentRuntime, EventJournal
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
    last_contact_at: str | None
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
    created_at: str | None
    updated_at: str | None
    started_at: str | None
    completed_at: str | None
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
class _JournalRef:
    event_id: str
    sequence: int


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
    def build() -> CockpitTrainingSummaryResponse:
        raw_nodes = coordinator.node_status()
        jobs = coordinator.list_jobs()
        adapters = registry.list()
        history = manager.history()
        journal = getattr(request.app.state, "event_journal", None)
        journal_by_sequence = _journal_by_sequence(journal)
        node_by_id = unique_by_id(raw_nodes, lambda item: _node_identifier(item))
        job_by_id = unique_by_id(jobs, lambda item: item.job_id)
        adapter_by_id = unique_by_id(adapters, lambda item: item.adapter_id)
        activation_by_adapter = unique_by_id(
            (item for item in history if item.action.startswith("activate") and item.adapter_id),
            lambda item: item.adapter_id or "",
        )
        rollback_by_adapter = unique_by_id(
            (item for item in history if item.action.startswith("rollback") and item.adapter_id),
            lambda item: item.adapter_id or "",
        )
        ordered_jobs = sorted(
            jobs,
            key=lambda item: (item.updated_at, item.created_at, item.job_id),
            reverse=True,
        )
        ordered_adapters = sorted(
            adapters,
            key=lambda item: (item.updated_at, item.created_at, item.adapter_id),
            reverse=True,
        )
        nodes = [_node_projection(node, settings) for node in node_by_id.values()]
        return CockpitTrainingSummaryResponse(
            node_count=len(node_by_id),
            online_node_count=sum(node.status == "online" for node in nodes),
            running_job_count=sum(item.status == TrainingJobStatus.RUNNING for item in jobs),
            failed_job_count=sum(item.status == TrainingJobStatus.FAILED for item in jobs),
            importing_job_count=sum(item.status == TrainingJobStatus.IMPORTING for item in jobs),
            active_adapter_count=sum(item.status.value == "active" for item in adapters),
            candidate_adapter_count=sum(item.status.value == "candidate" for item in adapters),
            nodes=nodes[:limit],
            jobs=[
                _job_projection(job if job_by_id.get(job.job_id) is job else None, adapter_by_id)
                for job in ordered_jobs[:limit]
            ],
            adapters=[
                _adapter_projection(
                    adapter if adapter_by_id.get(adapter.adapter_id) is adapter else None,
                    job_by_id,
                    activation_by_adapter,
                    rollback_by_adapter,
                    journal_by_sequence,
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


def _node_projection(
    node: dict[str, Any], settings: Settings
) -> CockpitTrainingNodeResponse:
    reachable = bool(node.get("reachable", False))
    observed_model = _safe_optional_text(node.get("model_id")) if reachable else None
    observed_revision = _safe_optional_text(node.get("model_revision")) if reachable else None
    expected = settings.deployment.training.remote_worker.expected_worker_model if (
        settings.deployment.training.remote_worker is not None
        and str(node.get("backend", "")) == "ssh"
    ) else settings.model
    gpu = node.get("gpu") if isinstance(node.get("gpu"), dict) else {}
    devices = gpu.get("devices") if isinstance(gpu, dict) else None
    gpu_name = None
    if isinstance(devices, list) and devices:
        gpu_name = _safe_optional_text(devices[0])
    status: Literal["online", "unavailable"] = "online" if reachable else "unavailable"
    return CockpitTrainingNodeResponse(
        node_id=_safe_identifier(node.get("node_id")) or "unavailable",
        role="worker" if str(node.get("backend", "")) == "ssh" else "inference",
        backend=_safe_text(node.get("backend")) or "unavailable",
        status=status,
        last_contact_at=_safe_optional_text(node.get("last_contact") or node.get("heartbeat")) if reachable else None,
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
    terminal = job.status in {TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED}
    return CockpitTrainingJobResponse(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        status=job.status.value,
        backend=_safe_optional_text(job.backend),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.phase_started_at if job.status in {TrainingJobStatus.RUNNING, TrainingJobStatus.SUCCEEDED, TrainingJobStatus.IMPORTING} else None,
        completed_at=job.updated_at if terminal else None,
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
        result_digest=None if adapter is None else _digest(adapter.adapter_hash),
    )


def _adapter_projection(
    adapter: AdapterEntry | None,
    job_by_id: dict[str, TrainingJob],
    activation_by_adapter: dict[str, AdapterActivationRecord],
    rollback_by_adapter: dict[str, AdapterActivationRecord],
    journal_by_sequence: dict[int, _JournalRef],
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
    activation = activation_by_adapter.get(adapter.adapter_id)
    if activation is not None and not _activation_matches(adapter, activation):
        activation = None
    rollback = rollback_by_adapter.get(adapter.adapter_id)
    if rollback is not None and not _activation_matches(adapter, rollback):
        rollback = None
    evaluation_id = adapter.real_model_behavioral_evaluation_id or adapter.behavioral_evaluation_id
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
        evaluation_status=_evaluation_status(adapter),
        approved=adapter.status.value in {"approved", "active", "archived"},
        active=adapter.status.value == "active",
        rollback_candidate=adapter.rollback_target_id is not None or adapter.rollout_state in {"canary", "canary_failed"},
        activation_event_id=_event_id(activation, journal_by_sequence),
        activation_event_sequence=None if activation is None else activation.activation_sequence,
        rollback_event_id=_event_id(rollback, journal_by_sequence),
        rollback_event_sequence=None if rollback is None else rollback.activation_sequence,
    )


def _job_adapter_matches(job: TrainingJob, adapter: AdapterEntry) -> bool:
    return (
        job.candidate_adapter_id == adapter.adapter_id
        and adapter.training_job_id in {None, job.job_id}
        and adapter.base_model == job.base_model_id
        and adapter.base_model_revision == job.base_model_revision
        and (adapter.training_node_id is None or adapter.training_node_id == job.worker_node_id)
    )


def _activation_matches(adapter: AdapterEntry, record: AdapterActivationRecord) -> bool:
    return record.adapter_id == adapter.adapter_id and record.adapter_hash == adapter.adapter_hash


def _event_id(
    record: AdapterActivationRecord | None, journal_by_sequence: dict[int, _JournalRef]
) -> str | None:
    if record is None:
        return None
    ref = journal_by_sequence.get(record.activation_sequence)
    return None if ref is None else ref.event_id


def _journal_by_sequence(journal: EventJournal | None) -> dict[int, _JournalRef]:
    if journal is None:
        return {}
    refs: dict[int, _JournalRef] = {}
    for record in journal.recent(500):
        sequence = record.processing_sequence
        if sequence is not None and record.event_type == AgentEventType.ADAPTER_UPDATE.value:
            refs[sequence] = _JournalRef(record.event_id, sequence)
    return refs


def unique_by_id(
    values: Iterable[RecordT], identifier: Callable[[RecordT], str]
) -> dict[str, RecordT]:
    grouped: dict[str, list[RecordT]] = {}
    for item in values:
        grouped.setdefault(identifier(item), []).append(item)
    return {record_id: items[0] for record_id, items in grouped.items() if len(items) == 1}


def _node_identifier(item: dict[str, Any]) -> str:
    return _safe_identifier(item.get("node_id")) or "unavailable"


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


def _evaluation_status(
    adapter: AdapterEntry,
) -> Literal["passed", "failed", "stale", "corrupt", "unavailable"]:
    state = adapter.real_model_behavioral_artifact_state or adapter.behavioral_artifact_state
    gate = adapter.real_model_behavioral_gate_passed
    if gate is None:
        gate = adapter.behavioral_gate_passed
    if state in {"quarantined"}:
        return "corrupt"
    if state in {"prepared", "finalized"}:
        return "stale"
    if state == "reconciled":
        return "passed" if gate is True else "failed"
    return "unavailable"

"""FastAPI dependency wiring."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fnmatch import fnmatchcase
from hmac import compare_digest
from ipaddress import ip_address
import os
from pathlib import Path
from threading import RLock
from typing import Callable, TypeVar
from uuid import uuid4

from fastapi import Header, HTTPException, Request, status

from kagya.api.observability import OperationalTelemetry, RuntimeEventLog
from kagya.actions import ActionExecutionLayer
from kagya.artifact_provenance import build_adapter_artifact_manifest
from kagya.config import Settings, TrainingBackendType, get_settings
from kagya.learning import (
    AdapterEntry,
    AdapterRegistry,
    AdapterStatus,
    AdapterRuntimeManager,
    RuntimeAdapterState,
    SleepCycleManager,
)
from kagya.memory import DualMemorySystem
from kagya.models import ModelProvider, load_model_provider
from kagya.outbox import Outbox
from kagya.runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    AgentStateStore,
    KagyaMainLoop,
    EmotionTimer,
    ExternalTransactionCoordinator,
    StateWAL,
    hash_snapshot,
    SchedulerBudget,
    SubjectScheduler,
)
from kagya.tools import ToolAuditLog, ToolExecutor, ToolRegistry
from kagya.training import (
    LocalTrainingBackend,
    MemoryConsolidator,
    SleepCoordinator,
    SSHTrainingBackend,
    TrainingBackend,
    CandidateArtifactImporter,
    DatasetGovernanceStore,
    TrainingBundleBuilder,
    TrainingJobRegistry,
)
from kagya.learning import QloraTrainer


T = TypeVar("T")
_dependency_lock = RLock()


class AdminRole(StrEnum):
    READ_ONLY = "read_only"
    APPROVAL_ONLY = "approval_only"
    FULL_ADMIN = "full_admin"


@dataclass(frozen=True)
class AdminActor:
    actor_id: str
    role: AdminRole
    reauthenticated: bool = False


_APPROVAL_PATHS = (
    "/api/adapters/*/approve",
    "/api/adapters/*/reject",
    "/api/memory/episodes/*/review",
    "/api/beliefs/*/resolve",
    "/api/beliefs/*/retract",
    "/api/beliefs/*/supersede",
    "/api/self-model/identity/proposals/*/resolve",
    "/api/actions/intents/*/approval",
    "/api/outbox/messages/*/responses",
)


def get_api_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def get_runtime_event_log(request: Request) -> RuntimeEventLog:
    event_log = getattr(request.app.state, "runtime_event_log", None)
    if event_log is None:
        event_log = RuntimeEventLog()
        request.app.state.runtime_event_log = event_log
    return event_log


def get_operational_telemetry(request: Request) -> OperationalTelemetry:
    telemetry = getattr(request.app.state, "operational_telemetry", None)
    if telemetry is None:
        settings = get_api_settings(request).observability
        telemetry = OperationalTelemetry(
            settings.metrics_path,
            settings.traces_path,
            max_series=settings.max_series,
            max_traces=settings.max_traces,
            enabled=settings.enabled,
        )
        request.app.state.operational_telemetry = telemetry
    return telemetry


def get_agent_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is not None:
        return runtime
    with _dependency_lock:
        runtime = getattr(request.app.state, "agent_runtime", None)
        if runtime is None:
            store = get_agent_state_store(request)
            snapshot = store.load(get_api_settings(request).emotion.baseline_surprisal)
            main_loop = get_main_loop(request)
            store.restore_into(main_loop, snapshot)

            def persist_completed(event: AgentEvent) -> str:
                saved = store.save(
                    store.capture(
                        request.app.state.main_loop,
                        _event_sequence(event.processing_sequence),
                    )
                )
                get_external_transaction_coordinator(request).finalize_event(
                    event.event_id, _event_sequence(event.processing_sequence)
                )
                return hash_snapshot(saved)

            def persist_failed(event: AgentEvent, exception: Exception) -> str | None:
                coordinator = get_external_transaction_coordinator(request)
                coordinator.orphan_event(event.event_id, type(exception).__name__)
                saved = store.save_failed_sequence(
                    _event_sequence(event.processing_sequence)
                )
                coordinator.compensate_event(event.event_id, type(exception).__name__)
                return None if saved is None else hash_snapshot(saved)

            runtime = AgentRuntime(
                queue_capacity=get_api_settings(request).api.agent_queue_capacity,
                event_recorder=get_runtime_event_log(request),
                initial_sequence=snapshot.last_processed_event_sequence,
                completion_hook=persist_completed,
                failure_hook=persist_failed,
                telemetry=get_operational_telemetry(request),
            )
            runtime.start()
            request.app.state.agent_runtime = runtime
            if get_api_settings(request).appraisal.timer_enabled:
                timer = EmotionTimer(
                    runtime,
                    lambda elapsed: request.app.state.main_loop.advance_time(elapsed),
                    interval_seconds=get_api_settings(
                        request
                    ).appraisal.timer_interval_seconds,
                )
                timer.start()
                request.app.state.emotion_timer = timer
    return runtime


def get_agent_state_store(request: Request) -> AgentStateStore:
    store = getattr(request.app.state, "agent_state_store", None)
    if store is None:
        store = AgentStateStore(
            get_api_settings(request).agent_state.path,
            get_runtime_event_log(request),
        )
        request.app.state.agent_state_store = store
    return store


def get_state_wal(request: Request) -> StateWAL:
    wal = getattr(request.app.state, "state_wal", None)
    if wal is None:
        wal = StateWAL(get_api_settings(request).agent_state_wal.path)
        request.app.state.state_wal = wal
    return wal


def get_subject_scheduler(request: Request) -> SubjectScheduler:
    scheduler = getattr(request.app.state, "subject_scheduler", None)
    if scheduler is None:
        runtime = get_agent_runtime(request)
        settings = get_api_settings(request).autonomy
        scheduler = SubjectScheduler(
            runtime,
            get_main_loop(request),
            budget=SchedulerBudget(
                max_events=settings.max_events_per_cycle,
                max_inferences=settings.max_inferences_per_cycle,
                max_wall_seconds=settings.max_wall_seconds_per_cycle,
            ),
            reevaluation_interval_seconds=settings.reevaluation_interval_seconds,
            telemetry=get_operational_telemetry(request),
        )
        request.app.state.subject_scheduler = scheduler
    return scheduler


def get_action_execution(request: Request) -> ActionExecutionLayer:
    execution = getattr(request.app.state, "action_execution", None)
    main_loop = get_main_loop(request)
    if execution is None or execution.main_loop is not main_loop:
        settings = get_api_settings(request).actions
        execution = ActionExecutionLayer(
            main_loop,
            document_root=settings.document_root,
            calendar_path=settings.calendar_path,
        )
        request.app.state.action_execution = execution
        main_loop.action_execution = execution
    return execution


def get_outbox(request: Request) -> Outbox:
    outbox = getattr(request.app.state, "outbox", None)
    main_loop = get_main_loop(request)
    if outbox is None or outbox.main_loop is not main_loop:
        settings = get_api_settings(request).outbox
        outbox = Outbox(
            main_loop,
            quiet_hours_start=settings.quiet_hours_start,
            quiet_hours_end=settings.quiet_hours_end,
            max_deliveries_per_hour=settings.max_deliveries_per_hour,
            event_recorder=get_runtime_event_log(request),
        )
        request.app.state.outbox = outbox
        main_loop.outbox = outbox
    return outbox


def _event_sequence(sequence: int | None) -> int:
    if sequence is None:
        raise RuntimeError("Agent event has no processing sequence")
    return sequence


def execute_agent_event(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    *,
    source: str,
    handler: Callable[[], T],
    payload: dict[str, object] | None = None,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> AgentEventOutcome[T]:
    try:
        return runtime.execute(
            event_type,
            source=source,
            handler=handler,
            payload=payload,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
    except AgentRuntimeQueueFull as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
    except AgentRuntimeStopped as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def require_admin(
    request: Request,
    x_kagya_admin_token: str | None = Header(default=None, alias="X-KAGYA-Admin-Token"),
) -> AdminActor:
    settings = get_api_settings(request)
    expected = os.getenv(settings.api.admin_token_env)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Admin token env var {settings.api.admin_token_env} is not configured",
        )
    if x_kagya_admin_token is None or not compare_digest(x_kagya_admin_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )
    auth = settings.api.admin_auth
    if not auth.enabled:
        actor = AdminActor("admin-token", AdminRole.FULL_ADMIN, True)
        _audit_admin_mutation(request, actor)
        return actor

    origin = request.headers.get("origin")
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site == "cross-site" or (fetch_site is not None and origin is None):
        raise HTTPException(status_code=403, detail="Cross-site admin request rejected")
    if origin is not None:
        if origin not in settings.api.cors_origins:
            raise HTTPException(
                status_code=403, detail="Admin request origin is not allowed"
            )
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            csrf_cookie = request.cookies.get(auth.csrf_cookie_name)
            csrf_header = request.headers.get(auth.csrf_header)
            if (
                request.cookies.get(auth.session_cookie_name) is None
                or csrf_cookie is None
                or csrf_header is None
                or not compare_digest(csrf_cookie, csrf_header)
            ):
                raise HTTPException(status_code=403, detail="Invalid admin CSRF token")

    actor_id = request.headers.get(auth.actor_header)
    role_value = request.headers.get(auth.role_header)
    if actor_id is None or role_value is None:
        if auth.allow_loopback_recovery and origin is None and _is_loopback(request):
            actor = AdminActor("local-recovery", AdminRole.FULL_ADMIN, True)
            _audit_admin_mutation(request, actor)
            return actor
        raise HTTPException(status_code=401, detail="Admin identity is required")
    try:
        role = AdminRole(role_value.strip().lower().replace("-", "_"))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid admin role") from exc
    actor_id = actor_id.strip()
    if not actor_id or len(actor_id) > 128:
        raise HTTPException(status_code=403, detail="Invalid admin actor")
    _enforce_role(request, role)

    reauthenticated = _recently_reauthenticated(request, settings)
    if _requires_reauthentication(request, settings) and not reauthenticated:
        raise HTTPException(
            status_code=403, detail="Recent re-authentication is required"
        )
    actor = AdminActor(actor_id, role, reauthenticated)
    _audit_admin_mutation(request, actor)
    return actor


def _enforce_role(request: Request, role: AdminRole) -> None:
    if role == AdminRole.FULL_ADMIN or request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    if role == AdminRole.APPROVAL_ONLY and any(
        fnmatchcase(request.url.path, pattern) for pattern in _APPROVAL_PATHS
    ):
        return
    raise HTTPException(
        status_code=403, detail="Admin role does not permit this operation"
    )


def _requires_reauthentication(request: Request, settings: Settings) -> bool:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return False
    return any(
        fnmatchcase(request.url.path, pattern)
        for pattern in settings.api.admin_auth.reauthentication_paths
    )


def _recently_reauthenticated(request: Request, settings: Settings) -> bool:
    value = request.headers.get(settings.api.admin_auth.reauthenticated_at_header)
    if value is None:
        return False
    try:
        timestamp = float(value)
    except ValueError:
        return False
    age = datetime.now(UTC).timestamp() - timestamp
    return -30 <= age <= settings.api.admin_auth.reauthentication_max_age_seconds


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client is not None else ""
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return host == "testclient"


def _audit_admin_mutation(request: Request, actor: AdminActor) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    journal = getattr(request.app.state, "event_journal", None)
    if journal is None:
        return
    journal.audit_admin_action(
        event_id=str(uuid4()),
        actor_id=actor.actor_id,
        actor_role=actor.role.value,
        target=f"{request.method} {request.url.path}",
        reauthenticated=actor.reauthenticated,
    )


def get_model_provider(request: Request) -> ModelProvider:
    provider = getattr(request.app.state, "model_provider", None)
    if provider is None:
        provider = load_model_provider(get_api_settings(request))
        request.app.state.model_provider = provider
    return provider


def sync_main_loop_to_active_adapter(request: Request) -> KagyaMainLoop:
    active_adapter = _get_active_adapter(request)
    provider = _get_runtime_model_provider(request, active_adapter)
    return _replace_main_loop(request, provider, active_adapter)


def _replace_main_loop(
    request: Request,
    provider: ModelProvider,
    active_adapter: AdapterEntry | None,
    activation_sequence_override: int | None = None,
) -> KagyaMainLoop:
    previous_loop = getattr(request.app.state, "main_loop", None)
    main_loop = KagyaMainLoop(
        get_api_settings(request),
        provider,
        get_memory_system(request),
        session_state=None if previous_loop is None else previous_loop.session_state,
        emotion_engine=None if previous_loop is None else previous_loop.emotion_engine,
        persistent_state=None
        if previous_loop is None
        else previous_loop.persistent_state,
        working_memory=None if previous_loop is None else previous_loop.working_memory,
        context_registry=None
        if previous_loop is None
        else previous_loop.context_registry,
        adapter_id=None if active_adapter is None else active_adapter.adapter_id,
        adapter_hash=None if active_adapter is None else active_adapter.adapter_hash,
        activation_sequence=(
            activation_sequence_override
            if activation_sequence_override is not None
            else None
            if active_adapter is None
            else active_adapter.activation_sequence
        ),
        telemetry=get_operational_telemetry(request),
    )
    request.app.state.main_loop = main_loop
    scheduler = getattr(request.app.state, "subject_scheduler", None)
    if scheduler is not None:
        scheduler.main_loop = main_loop
    execution = getattr(request.app.state, "action_execution", None)
    if execution is not None:
        execution.main_loop = main_loop
        main_loop.action_execution = execution
    return main_loop


def get_adapter_runtime_manager(request: Request) -> AdapterRuntimeManager:
    manager = getattr(request.app.state, "adapter_runtime_manager", None)
    if manager is not None:
        return manager
    settings = get_api_settings(request)
    registry = get_adapter_registry(request)

    def load(entry: AdapterEntry | None) -> ModelProvider:
        manifest = (
            None
            if entry is None
            else build_adapter_artifact_manifest(
                Path(entry.path),
                base_model_name=entry.base_model,
                base_model_revision=entry.base_model_revision,
            )
        )
        provider = load_model_provider(
            settings,
            adapter_path=None if entry is None else entry.path,
            allow_archived_adapter=(
                entry is not None and entry.status == AdapterStatus.ARCHIVED
            ),
            expected_adapter_hash=None if entry is None else entry.adapter_hash,
            expected_adapter_manifest=manifest,
        )
        if entry is not None and settings.model.provider == "dummy":
            assert manifest is not None
            provider.adapter_artifact_manifest = manifest  # type: ignore[attr-defined]
            provider.adapter_artifact_manifest_hash = manifest.sha256  # type: ignore[attr-defined]
            provider.adapter_snapshot_manifest_hash = manifest.sha256  # type: ignore[attr-defined]
            provider.adapter_snapshot_hash = entry.adapter_hash  # type: ignore[attr-defined]
        return provider

    def switch(
        provider: ModelProvider, entry: AdapterEntry | None, sequence: int | None
    ) -> None:
        request.app.state.model_provider = provider
        request.app.state.model_provider_adapter_id = (
            None if entry is None else entry.adapter_id
        )
        _replace_main_loop(
            request, provider, entry, activation_sequence_override=sequence
        )

    def snapshot() -> RuntimeAdapterState:
        loop = getattr(request.app.state, "main_loop", None)
        if loop is None:
            loop = get_main_loop(request)
        return RuntimeAdapterState(
            adapter_id=loop.adapter_id,
            adapter_hash=loop.adapter_hash,
            activation_sequence=loop.activation_sequence,
            provider=loop.provider,
        )

    manager = AdapterRuntimeManager(
        registry,
        provider_loader=load,
        runtime_switch=switch,
        runtime_snapshot=snapshot,
        history_path=settings.adapter_registry.path.with_name(
            f"{settings.adapter_registry.path.stem}_activations.json"
        ),
    )
    request.app.state.adapter_runtime_manager = manager
    return manager


def get_memory_system(request: Request) -> DualMemorySystem:
    memory = getattr(request.app.state, "memory_system", None)
    if memory is None:
        memory = DualMemorySystem(get_api_settings(request))
        request.app.state.memory_system = memory
    return memory


def get_external_transaction_coordinator(
    request: Request,
) -> ExternalTransactionCoordinator:
    coordinator = getattr(request.app.state, "external_transaction_coordinator", None)
    if coordinator is None:
        coordinator = ExternalTransactionCoordinator([get_memory_system(request)])
        request.app.state.external_transaction_coordinator = coordinator
    return coordinator


def get_adapter_registry(request: Request) -> AdapterRegistry:
    registry = getattr(request.app.state, "adapter_registry", None)
    if registry is None:
        registry = AdapterRegistry(get_api_settings(request))
        request.app.state.adapter_registry = registry
    return registry


def get_tool_registry(request: Request) -> ToolRegistry:
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        registry = ToolRegistry(get_api_settings(request).tools.path)
        request.app.state.tool_registry = registry
    return registry


def get_tool_executor(request: Request) -> ToolExecutor:
    executor = getattr(request.app.state, "tool_executor", None)
    if executor is None:
        settings = get_api_settings(request)
        executor = ToolExecutor(
            get_tool_registry(request),
            audit_log_store=ToolAuditLog(settings.tools.audit_path),
        )
        request.app.state.tool_executor = executor
    return executor


def get_main_loop(request: Request) -> KagyaMainLoop:
    main_loop = getattr(request.app.state, "main_loop", None)
    active_adapter = _get_active_adapter(request)
    active_adapter_id = None if active_adapter is None else active_adapter.adapter_id
    if main_loop is None or main_loop.adapter_id != active_adapter_id:
        with _dependency_lock:
            main_loop = getattr(request.app.state, "main_loop", None)
            if main_loop is None or main_loop.adapter_id != active_adapter_id:
                main_loop = sync_main_loop_to_active_adapter(request)
    return main_loop


def get_sleep_cycle_manager(request: Request) -> SleepCycleManager:
    return SleepCycleManager(
        get_api_settings(request),
        get_memory_system(request),
        get_model_provider(request),
        get_adapter_registry(request),
    )


def get_sleep_coordinator(request: Request) -> SleepCoordinator:
    coordinator = getattr(request.app.state, "sleep_coordinator", None)
    if coordinator is not None:
        return coordinator
    with _dependency_lock:
        coordinator = getattr(request.app.state, "sleep_coordinator", None)
        if coordinator is None:
            settings = get_api_settings(request)
            runtime = get_agent_runtime(request)
            if settings.deployment.training.backend == TrainingBackendType.SSH:
                remote = settings.deployment.training.remote_worker
                if remote is None:
                    raise RuntimeError(
                        "SSH training backend requires remote worker settings"
                    )
                backend: TrainingBackend = SSHTrainingBackend(
                    remote, settings.sleep.training_artifact_directory
                )
            else:
                backend = LocalTrainingBackend(
                    QloraTrainer(settings, get_adapter_registry(request))
                )
            coordinator = SleepCoordinator(
                settings,
                MemoryConsolidator(
                    settings, get_memory_system(request), get_model_provider(request)
                ),
                TrainingBundleBuilder(
                    settings,
                    get_adapter_registry(request),
                    get_dataset_governance(request),
                ),
                TrainingJobRegistry(settings.sleep.job_registry_path),
                backend,
                get_adapter_registry(request),
                subject_executor=lambda source, handler: (
                    runtime.execute(
                        AgentEventType.SLEEP,
                        source=source,
                        handler=handler,
                    ).value
                ),
                candidate_importer=(
                    CandidateArtifactImporter(settings, get_adapter_registry(request))
                    if settings.deployment.training.backend == TrainingBackendType.SSH
                    else None
                ),
            )
            request.app.state.sleep_coordinator = coordinator
    return coordinator


def get_dataset_governance(request: Request) -> DatasetGovernanceStore:
    store = getattr(request.app.state, "dataset_governance", None)
    if store is None:
        with _dependency_lock:
            store = getattr(request.app.state, "dataset_governance", None)
            if store is None:
                settings = get_api_settings(request)
                store = DatasetGovernanceStore(
                    settings.sleep.training_artifact_directory / "datasets"
                )
                request.app.state.dataset_governance = store
    return store


def _get_active_adapter(request: Request) -> AdapterEntry | None:
    return next(
        (
            entry
            for entry in get_adapter_registry(request).list()
            if entry.status == AdapterStatus.ACTIVE
        ),
        None,
    )


def _get_runtime_model_provider(
    request: Request, active_adapter: AdapterEntry | None
) -> ModelProvider:
    settings = get_api_settings(request)
    provider_adapter_id = getattr(request.app.state, "model_provider_adapter_id", None)
    if settings.model.provider.lower() == "transformers" and active_adapter is not None:
        if (
            getattr(request.app.state, "model_provider", None) is None
            or provider_adapter_id != active_adapter.adapter_id
        ):
            manifest = build_adapter_artifact_manifest(
                Path(active_adapter.path),
                base_model_name=active_adapter.base_model,
                base_model_revision=active_adapter.base_model_revision,
            )
            request.app.state.model_provider = load_model_provider(
                settings,
                adapter_path=active_adapter.path,
                expected_adapter_hash=active_adapter.adapter_hash,
                expected_adapter_manifest=manifest,
            )
            request.app.state.model_provider_adapter_id = active_adapter.adapter_id
        return request.app.state.model_provider

    provider = get_model_provider(request)
    request.app.state.model_provider_adapter_id = None
    return provider

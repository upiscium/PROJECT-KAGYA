"""Administrative agent-state snapshot operations."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_agent_state_store,
    get_api_settings,
    get_external_transaction_coordinator,
    get_adapter_runtime_manager,
    get_dataset_governance,
    get_main_loop,
    get_sleep_coordinator,
    get_state_wal,
    get_tool_executor,
    get_tool_registry,
)
from kagya.config import Settings
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentStateSnapshot,
    AgentStateStore,
    ExternalReconciliationReport,
    ExternalRestoreDiff,
    ExternalTransactionCoordinator,
    ExternalTransactionRecord,
    StateDryRun,
    StateReconstruction,
    StateWAL,
    StateWalIntegrityError,
)
from kagya.runtime.agent_state import default_agent_state_snapshot
from kagya.security.backup import (
    BackupError,
    BackupManager,
    BackupStatus,
    RestorePreview,
)
from kagya.security.crypto import EncryptionError


router = APIRouter(prefix="/api/state", tags=["state"])


class BackupCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_backup_id: str | None = None


class BackupRestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_backup_id: str
    expected_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkingMemorySummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    item_capacity: int = Field(gt=0)
    token_capacity: int = Field(gt=0)


@router.get("/working-memory", response_model=WorkingMemorySummaryResponse)
def working_memory_summary(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> WorkingMemorySummaryResponse:
    working_memory = get_main_loop(request).working_memory
    return execute_agent_event(
        runtime,
        AgentEventType.MEMORY_READ,
        source="api.state.working_memory",
        handler=lambda: WorkingMemorySummaryResponse(
            item_count=len(working_memory.items),
            token_count=working_memory.token_count,
            item_capacity=working_memory.item_capacity,
            token_capacity=working_memory.token_capacity,
        ),
    ).value


@router.post("/snapshot", response_model=AgentStateSnapshot)
def create_snapshot(
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
) -> AgentStateSnapshot:
    execute_agent_event(
        runtime,
        AgentEventType.STATE_SNAPSHOT,
        source="api.state.snapshot",
        handler=lambda: None,
    )
    return _current_snapshot(store)


@router.get("/export", response_model=AgentStateSnapshot)
def export_snapshot(
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
) -> AgentStateSnapshot:
    execute_agent_event(
        runtime,
        AgentEventType.STATE_EXPORT,
        source="api.state.export",
        handler=lambda: None,
    )
    return _current_snapshot(store)


@router.post("/restore", response_model=AgentStateSnapshot)
def restore_snapshot(
    snapshot: AgentStateSnapshot,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
) -> AgentStateSnapshot:
    execute_agent_event(
        runtime,
        AgentEventType.STATE_RESTORE,
        source="api.state.restore",
        handler=lambda: store.restore_into(get_main_loop(request), snapshot),
    )
    return _current_snapshot(store)


@router.post("/reset", response_model=AgentStateSnapshot)
def reset_snapshot(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
    settings: Settings = Depends(get_api_settings),
) -> AgentStateSnapshot:
    def reset() -> None:
        main_loop = get_main_loop(request)
        store.restore_into(
            main_loop,
            default_agent_state_snapshot(settings.emotion.baseline_surprisal),
        )
        main_loop.session_state.turns.clear()
        main_loop.working_memory.clear()
        main_loop.context_registry.clear()

    execute_agent_event(
        runtime,
        AgentEventType.STATE_RESET,
        source="api.state.reset",
        handler=reset,
    )
    return _current_snapshot(store)


@router.get("/reconstruct/{sequence}", response_model=StateReconstruction)
def reconstruct_snapshot(
    sequence: int,
    wal: StateWAL = Depends(get_state_wal),
) -> StateReconstruction:
    try:
        return wal.reconstruct(sequence)
    except StateWalIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/restore/{sequence}/dry-run", response_model=StateDryRun)
def dry_run_restore(
    sequence: int,
    store: AgentStateStore = Depends(get_agent_state_store),
    wal: StateWAL = Depends(get_state_wal),
) -> StateDryRun:
    try:
        return wal.dry_run(_current_snapshot(store), sequence)
    except StateWalIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.get("/external-transactions", response_model=list[ExternalTransactionRecord])
def external_transaction_audit(
    coordinator: ExternalTransactionCoordinator = Depends(
        get_external_transaction_coordinator
    ),
) -> list[ExternalTransactionRecord]:
    return coordinator.records()


@router.post(
    "/external-transactions/reconcile",
    response_model=ExternalReconciliationReport,
)
def reconcile_external_transactions(
    request: Request,
    coordinator: ExternalTransactionCoordinator = Depends(
        get_external_transaction_coordinator
    ),
) -> ExternalReconciliationReport:
    return coordinator.reconcile(request.app.state.event_journal.verify())


@router.get("/restore/{sequence}/external-diff", response_model=ExternalRestoreDiff)
def external_restore_diff(
    sequence: int,
    wal: StateWAL = Depends(get_state_wal),
    coordinator: ExternalTransactionCoordinator = Depends(
        get_external_transaction_coordinator
    ),
) -> ExternalRestoreDiff:
    try:
        wal.reconstruct(sequence)
    except StateWalIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return coordinator.restore_diff(sequence)


@router.post("/restore/{sequence}", response_model=AgentStateSnapshot)
def restore_point_in_time(
    sequence: int,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
    wal: StateWAL = Depends(get_state_wal),
) -> AgentStateSnapshot:
    def restore() -> None:
        target = wal.reconstruct(sequence)
        store.restore_into(get_main_loop(request), target.snapshot)

    try:
        execute_agent_event(
            runtime,
            AgentEventType.STATE_POINT_IN_TIME_RESTORE,
            source="api.state.point_in_time_restore",
            handler=restore,
        )
    except StateWalIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return _current_snapshot(store)


@router.post("/backups", response_model=RestorePreview)
def create_encrypted_backup(
    body: BackupCreateRequest,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    manager = BackupManager(settings)
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.BACKUP_CREATE,
            source="api.state.backup.create",
            handler=lambda: manager.create(base_backup_id=body.base_backup_id),
        ).value
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/backups", response_model=list[BackupStatus])
def list_encrypted_backups(
    limit: int = 50,
    settings: Settings = Depends(get_api_settings),
) -> list[BackupStatus]:
    try:
        return BackupManager(settings).list(limit)
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/backups/{backup_id}/preview", response_model=RestorePreview)
def preview_encrypted_backup(
    backup_id: str,
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    try:
        return BackupManager(settings).preview(backup_id)
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/verify", response_model=RestorePreview)
def verify_encrypted_backup(
    backup_id: str,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    manager = BackupManager(settings)
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.BACKUP_VERIFY,
            source="api.state.backup.verify",
            handler=lambda: manager.verify(backup_id),
        ).value
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/rotate", response_model=RestorePreview)
def rotate_encrypted_backup(
    backup_id: str,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    manager = BackupManager(settings)
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.BACKUP_ROTATE,
            source="api.state.backup.rotate",
            handler=lambda: manager.rotate(backup_id),
        ).value
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/restore", response_model=RestorePreview)
def commit_encrypted_restore(
    backup_id: str,
    body: BackupRestoreRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    if settings.failover.enabled:
        raise HTTPException(
            status_code=409,
            detail="online backup restore is unavailable while subject failover is enabled",
        )
    if body.expected_backup_id != backup_id:
        raise HTTPException(status_code=409, detail="expected backup ID does not match")
    manager = BackupManager(settings)
    initialized_caches = {
        name
        for name in (
            "sleep_coordinator",
            "dataset_governance",
            "adapter_runtime_manager",
            "tool_registry",
            "tool_executor",
        )
        if getattr(request.app.state, name, None) is not None
    }
    # Serialize all accepted work before the offline-style authoritative swap.
    execute_agent_event(
        runtime,
        AgentEventType.BACKUP_RESTORE,
        source="api.state.backup.restore.prepare",
        handler=lambda: manager.verify(backup_id),
    )
    try:
        restored = manager.restore(
            backup_id,
            expected_manifest_hash=body.expected_manifest_hash,
            prepare=lambda _root: _teardown_subject_runtime(request),
            after_publish=lambda: _build_subject_runtime_offline(
                request, settings, initialized_caches
            ),
            activate=lambda: _activate_subject_runtime(request),
            after_rollback=lambda: _reset_build_and_activate_subject_runtime(
                request, settings, initialized_caches
            ),
        )
        return restored
    except Exception as exc:
        if isinstance(exc, (BackupError, EncryptionError)):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise


def _build_subject_runtime_offline(
    request: Request,
    settings: Settings,
    initialized_caches: set[str],
) -> None:
    from kagya.api.server import _build_subject_runtime

    _build_subject_runtime(
        request.app, settings, reconcile=False, allow_restore_marker=True
    )
    if "dataset_governance" in initialized_caches:
        get_dataset_governance(request)
    if "sleep_coordinator" in initialized_caches:
        get_sleep_coordinator(request)
    if "adapter_runtime_manager" in initialized_caches:
        get_adapter_runtime_manager(request)
    if "tool_registry" in initialized_caches:
        get_tool_registry(request)
    if "tool_executor" in initialized_caches:
        get_tool_executor(request)


def _activate_subject_runtime(request: Request) -> None:
    from kagya.api.server import _activate_subject_runtime as activate

    try:
        activate(request.app)
    except Exception:
        _teardown_subject_runtime(request)
        raise


def _teardown_subject_runtime(request: Request) -> None:
    from kagya.api.server import teardown_subject_runtime

    teardown_subject_runtime(request.app)


def _reset_build_and_activate_subject_runtime(
    request: Request, settings: Settings, initialized_caches: set[str]
) -> None:
    _teardown_subject_runtime(request)
    _build_subject_runtime_offline(request, settings, initialized_caches)
    _activate_subject_runtime(request)


def _current_snapshot(store: AgentStateStore) -> AgentStateSnapshot:
    if store.last_snapshot is None:
        raise RuntimeError("Agent state snapshot is unavailable")
    return store.last_snapshot
